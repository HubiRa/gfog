"""GFog negative-control benchmark on SHA-256 proof-of-work style search.

The objective behaves like a random oracle: candidates are byte strings, and
the score is the normalized leading 64 bits of double-SHA256(prefix || bytes).
Lower is better. A learning method should not beat random search consistently.

The script also includes intentionally weak toy hash modes. Those are not
cryptographic; they are controls for "structure exists" cases.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gfog.buffer import Buffer  # noqa: E402
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig  # noqa: E402
from gfog.models import MLP  # noqa: E402
from gfog.opt import QuantileRankedDefaultOpt, components  # noqa: E402
from gfog.opt.latents_sampler import LatentSamplerLambda  # noqa: E402


def zeropower_via_newton_schulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the zeroth power/orthogonal factor of a 2D gradient."""
    if g.ndim != 2:
        raise ValueError(
            f"Muon zeropower expects a 2D tensor, got shape={tuple(g.shape)}"
        )

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T

    x = x / torch.clamp(x.norm(), min=1e-12)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transpose:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Small experimental Muon optimizer for matrix-heavy MLPs."""

    def __init__(
        self,
        parameters,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                update = grad.add(buf, alpha=beta) if nesterov else buf
                if update.ndim == 2:
                    update_2d = zeropower_via_newton_schulz5(update, steps=ns_steps)
                    scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                    p.add_(update_2d, alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr)
        return loss


def make_torch_optimizer(name: str, parameters, *, lr: float, momentum: float):
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if name == "muon":
        return Muon(parameters, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


class ByteGenerator(nn.Module):
    """Unconstrained latent-to-byte-coordinate generator."""

    def __init__(
        self,
        *,
        latent_dim: int,
        byte_dim: int,
        hidden_dim: int,
        output_scale: float,
        output_transform: str,
        output_temperature: float,
        output_bias_init: float,
    ) -> None:
        super().__init__()
        self.output_scale = output_scale
        if output_temperature <= 0:
            raise ValueError(
                f"output_temperature must be > 0, got {output_temperature}"
            )
        self.output_temperature = output_temperature
        if output_transform not in {"raw", "softplus_l2"}:
            raise ValueError(
                "output_transform must be one of raw, softplus_l2; "
                f"got {output_transform}"
            )
        self.output_transform = output_transform
        self.net = MLP(
            input_dim=latent_dim,
            output_dim=byte_dim,
            hidden_dims=[hidden_dim, hidden_dim],
        )
        if output_transform == "softplus_l2":
            with torch.no_grad():
                self.net.layers[-1].bias.fill_(output_bias_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        if self.output_transform == "softplus_l2":
            out = torch.nn.functional.softplus(self.output_temperature * out)
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return self.output_scale * out


class ShaPowObjective:
    """Proof-of-work style score over quantized candidate byte vectors."""

    def __init__(
        self,
        *,
        prefix: bytes,
        byte_dim: int,
        candidate_encoding: str,
        binhead_input_is_positive: bool,
        hash_mode: str,
        toy_rounds: int,
    ) -> None:
        self.prefix = prefix
        self.byte_dim = byte_dim
        if candidate_encoding not in {"raw_bytes", "binhead_bits"}:
            raise ValueError(
                "candidate_encoding must be one of raw_bytes, binhead_bits; "
                f"got {candidate_encoding}"
            )
        self.candidate_encoding = candidate_encoding
        self.binhead_input_is_positive = binhead_input_is_positive
        self.input_dim = byte_dim if candidate_encoding == "raw_bytes" else byte_dim * 8
        if hash_mode not in {"sha256", "toy_arx", "byte_linear"}:
            raise ValueError(
                "hash_mode must be one of sha256, toy_arx, byte_linear; "
                f"got {hash_mode}"
            )
        if toy_rounds < 0:
            raise ValueError(f"toy_rounds must be non-negative, got {toy_rounds}")
        self.hash_mode = hash_mode
        self.toy_rounds = toy_rounds
        self.evaluation_count = 0
        self.seen_candidates: set[bytes] = set()

    def _to_bytes(self, x: np.ndarray) -> list[bytes]:
        if self.candidate_encoding == "raw_bytes":
            raw = np.rint(x).astype(np.int64).reshape(-1, self.byte_dim)
            wrapped = np.mod(raw, 256).astype(np.uint8)
            return [row.tobytes() for row in wrapped]

        raw_bits = np.asarray(x, dtype=np.float32).reshape(-1, self.input_dim)
        bits = self._project_positive_vector_to_binary(
            raw_bits,
            input_is_positive=self.binhead_input_is_positive,
        )
        packed = np.packbits(bits.astype(np.uint8), axis=1, bitorder="big")
        return [row.tobytes() for row in packed]

    @staticmethod
    def _project_positive_vector_to_binary(
        x: np.ndarray,
        *,
        input_is_positive: bool,
    ) -> np.ndarray:
        """Project continuous logits to binary subsets using the BinHead rule."""
        if input_is_positive:
            positive = np.maximum(x, 0.0) + 1e-8
        else:
            positive = np.logaddexp(x, 0.0) + 1e-8
        norms = np.linalg.norm(positive, axis=1, keepdims=True)
        positive = positive / np.maximum(norms, 1e-12)
        order = np.argsort(-positive, axis=1)
        sorted_values = np.take_along_axis(positive, order, axis=1)
        k = np.arange(1, positive.shape[1] + 1, dtype=np.float32)
        scores = np.cumsum(sorted_values, axis=1) / np.sqrt(k)[None, :]
        best_k = np.argmax(scores, axis=1) + 1
        bits = np.zeros_like(positive, dtype=np.uint8)
        for row_idx, row_k in enumerate(best_k):
            bits[row_idx, order[row_idx, :row_k]] = 1
        return bits

    def score_bytes(self, candidates: list[bytes]) -> np.ndarray:
        scores = np.empty(len(candidates), dtype=np.float64)
        for idx, candidate in enumerate(candidates):
            if self.hash_mode == "sha256":
                leading = self._score_sha256_u64(candidate)
            elif self.hash_mode == "toy_arx":
                leading = self._score_toy_arx_u64(candidate)
            else:
                leading = self._score_byte_linear_u64(candidate)
            scores[idx] = leading / float(1 << 64)
        return scores

    def _score_sha256_u64(self, candidate: bytes) -> int:
        digest = hashlib.sha256(
            hashlib.sha256(self.prefix + candidate).digest()
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _score_toy_arx_u64(self, candidate: bytes) -> int:
        """A deliberately weak low-round add-rotate-xor mixer."""
        mask = (1 << 64) - 1
        data = self.prefix + candidate
        a = 0x243F6A8885A308D3
        b = 0x13198A2E03707344
        c = 0xA4093822299F31D0
        for offset, byte in enumerate(data):
            shift = (offset % 8) * 8
            a ^= (byte + 1) << shift
            b = (b + ((byte + offset + 1) * 0x9E3779B185EBCA87)) & mask
            if self.toy_rounds == 0:
                continue
            for _ in range(self.toy_rounds):
                a = (a + b) & mask
                b ^= ((a << 13) | (a >> 51)) & mask
                c = (c + b + byte) & mask
                a ^= ((c << 17) | (c >> 47)) & mask
        return (a ^ b ^ c) & mask

    def _score_byte_linear_u64(self, candidate: bytes) -> int:
        """Very weak structured objective disguised as a 64-bit score."""
        mask = (1 << 64) - 1
        data = self.prefix + candidate
        total = 0
        for idx, byte in enumerate(data):
            weight = ((idx + 1) * 0x9E3779B1) & mask
            total = (total + weight * byte) & mask
        return total

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[float]:
        if isinstance(theta, torch.Tensor):
            x_np = theta.detach().to("cpu", torch.float32).numpy()
        else:
            x_np = np.asarray(theta, dtype=np.float32)
        candidates = self._to_bytes(x_np)
        self.evaluation_count += len(candidates)
        self.seen_candidates.update(candidates)
        return self.score_bytes(candidates).astype(float).tolist()


def record_history(opt: QuantileRankedDefaultOpt, iteration: int) -> dict[str, float]:
    values = np.asarray([row[-1] for row in opt.buffer.B.get_sorted_values()])
    return {
        "iteration": float(iteration),
        "eval_count": float(
            opt.buffer.B.buffer_size + iteration * opt.components.batch_size
        ),
        "best": float(values[0]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def plot_history(
    gfog_history: list[dict[str, float]],
    random_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for history, label in [(gfog_history, "GFog"), (random_history, "Random")]:
        evals = np.asarray([row["eval_count"] for row in history])
        best = np.asarray([row["best"] for row in history])
        ax.plot(evals, best, label=f"{label} best", linewidth=2)
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("normalized leading hash value, lower is better")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_optimizer(args: argparse.Namespace, objective: ShaPowObjective):
    device = torch.device("cpu")
    g = ByteGenerator(
        latent_dim=args.latent_dim,
        byte_dim=objective.input_dim,
        hidden_dim=args.hidden_dim,
        output_scale=args.generator_output_scale,
        output_transform=args.generator_output_transform,
        output_temperature=args.generator_output_temperature,
        output_bias_init=args.generator_output_bias_init,
    ).to(device)
    d = MLP(
        input_dim=objective.input_dim,
        output_dim=1,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
        use_spectral_norm=True,
    ).to(device)
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
    )
    curiosity_loss = None
    if args.curiosity > 0:
        curiosity_loss = WangIsolaUniformity(
            WangIsolaUniformityConfig(
                t=args.curiosity_t,
                use_buffer=args.curiosity_reference == "buffer",
                weight=args.curiosity,
            ),
            buffer=buffer.B,
        )
    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
        latent_dim=args.latent_dim,
        optimizerG=make_torch_optimizer(
            args.g_optimizer,
            g.parameters(),
            lr=args.g_lr,
            momentum=args.optimizer_momentum,
        ),
        optimizerD=make_torch_optimizer(
            args.d_optimizer,
            d.parameters(),
            lr=args.d_lr,
            momentum=args.optimizer_momentum,
        ),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d),
            b=args.batch_size,
            d=args.latent_dim,
        ),
        device=device,
        dtype=torch.float32,
    )
    opt_components = components.OptComponents(
        fn=components.Fn(
            f=objective,
            input_dim=objective.input_dim,
            device=device,
            dtype=torch.float32,
        ),
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=1,
        elite_sampling="random_top_k",
        elite_pool_size=args.buffer_multiplier * args.batch_size,
    )
    return QuantileRankedDefaultOpt(
        opt_components,
        ranker_list_size=args.ranker_list_size,
        ranker_steps=1,
        ranker_sample_pool_size=args.ranker_sample_pool_size,
        ranker_sample_mode="random_top_pool",
        ranker_weight=1.0,
        ranker_target_curve="exp",
        ranker_tau=args.ranker_tau,
        ranker_target_scope="local",
    )


def random_search(
    *,
    objective: ShaPowObjective,
    budget: int,
    batch_size: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    best = float("inf")
    history: list[dict[str, float]] = []
    eval_count = 0
    seen_candidates: set[bytes] = set()
    while eval_count < budget:
        n = min(batch_size, budget - eval_count)
        if objective.candidate_encoding == "raw_bytes":
            candidates = [
                row.tobytes()
                for row in rng.integers(
                    0, 256, size=(n, objective.byte_dim), dtype=np.uint8
                )
            ]
        else:
            candidates = [
                row.tobytes()
                for row in rng.integers(
                    0, 256, size=(n, objective.byte_dim), dtype=np.uint8
                )
            ]
        seen_candidates.update(candidates)
        scores = objective.score_bytes(candidates)
        eval_count += n
        best = min(best, float(scores.min()))
        if eval_count == budget or eval_count % max(batch_size, budget // 40) == 0:
            history.append(
                {
                    "iteration": float(eval_count // batch_size),
                    "eval_count": float(eval_count),
                    "best": best,
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "p10": float(np.percentile(scores, 10)),
                    "p90": float(np.percentile(scores, 90)),
                }
            )
    objective.seen_candidates = seen_candidates
    return best, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHA proof-of-work negative-control benchmark"
    )
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_multiplier", type=int, default=8)
    parser.add_argument("--byte_dim", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--generator_output_scale", type=float, default=1.0)
    parser.add_argument(
        "--generator_output_transform",
        choices=["raw", "softplus_l2"],
        default="raw",
    )
    parser.add_argument("--generator_output_temperature", type=float, default=1.0)
    parser.add_argument("--generator_output_bias_init", type=float, default=0.0)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--g_lr", type=float, default=0.003)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--curiosity", type=float, default=0.0)
    parser.add_argument("--curiosity_t", type=float, default=2.0)
    parser.add_argument(
        "--curiosity_reference",
        choices=["batch", "buffer"],
        default="batch",
    )
    parser.add_argument("--prefix", type=str, default="gfog-proof-of-work")
    parser.add_argument(
        "--candidate_encoding",
        choices=["raw_bytes", "binhead_bits"],
        default="raw_bytes",
        help="How f decodes G output before hashing.",
    )
    parser.add_argument(
        "--hash_mode",
        choices=["sha256", "toy_arx", "byte_linear"],
        default="sha256",
    )
    parser.add_argument(
        "--toy_rounds",
        type=int,
        default=2,
        help="Number of ARX rounds per byte for --hash_mode toy_arx.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/crypto_pow_sha")
    )
    parser.add_argument("--history_interval", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    objective = ShaPowObjective(
        prefix=args.prefix.encode("utf-8"),
        byte_dim=args.byte_dim,
        candidate_encoding=args.candidate_encoding,
        binhead_input_is_positive=args.generator_output_transform == "softplus_l2",
        hash_mode=args.hash_mode,
        toy_rounds=args.toy_rounds,
    )
    opt = build_optimizer(args, objective)
    history = [record_history(opt, 0)]
    for iteration in range(1, args.n_iter + 1):
        opt.step()
        if iteration % args.history_interval == 0 or iteration == args.n_iter:
            history.append(record_history(opt, iteration))

    budget = int(objective.evaluation_count)
    random_objective = ShaPowObjective(
        prefix=args.prefix.encode("utf-8"),
        byte_dim=args.byte_dim,
        candidate_encoding=args.candidate_encoding,
        binhead_input_is_positive=False,
        hash_mode=args.hash_mode,
        toy_rounds=args.toy_rounds,
    )
    random_best, random_history = random_search(
        objective=random_objective,
        budget=budget,
        batch_size=args.batch_size,
        seed=args.seed + 10_000,
    )
    best = history[-1]["best"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "history.png"
    plot_history(history, random_history, plot_path)
    np.savez_compressed(
        args.output_dir / "sha_pow_results.npz",
        gfog_history=np.asarray(
            [
                [
                    row[k]
                    for k in (
                        "iteration",
                        "eval_count",
                        "best",
                        "mean",
                        "median",
                        "p10",
                        "p90",
                    )
                ]
                for row in history
            ],
            dtype=np.float64,
        ),
        random_history=np.asarray(
            [
                [
                    row[k]
                    for k in (
                        "iteration",
                        "eval_count",
                        "best",
                        "mean",
                        "median",
                        "p10",
                        "p90",
                    )
                ]
                for row in random_history
            ],
            dtype=np.float64,
        ),
        history_columns=np.asarray(
            ["iteration", "eval_count", "best", "mean", "median", "p10", "p90"]
        ),
        gfog_best=np.asarray([best], dtype=np.float64),
        random_best=np.asarray([random_best], dtype=np.float64),
        budget=np.asarray([budget], dtype=np.int64),
        gfog_unique_candidates=np.asarray(
            [len(objective.seen_candidates)], dtype=np.int64
        ),
        random_unique_candidates=np.asarray(
            [len(random_objective.seen_candidates)], dtype=np.int64
        ),
        byte_dim=np.asarray([args.byte_dim], dtype=np.int32),
        seed=np.asarray([args.seed], dtype=np.int32),
        prefix=np.asarray([args.prefix]),
        hash_mode=np.asarray([args.hash_mode]),
        toy_rounds=np.asarray([args.toy_rounds], dtype=np.int32),
        candidate_encoding=np.asarray([args.candidate_encoding]),
        input_dim=np.asarray([objective.input_dim], dtype=np.int32),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        generator_output_scale=np.asarray(
            [args.generator_output_scale], dtype=np.float32
        ),
        generator_output_transform=np.asarray([args.generator_output_transform]),
        generator_output_temperature=np.asarray(
            [args.generator_output_temperature], dtype=np.float32
        ),
        generator_output_bias_init=np.asarray(
            [args.generator_output_bias_init], dtype=np.float32
        ),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        curiosity_t=np.asarray([args.curiosity_t], dtype=np.float32),
        curiosity_reference=np.asarray([args.curiosity_reference]),
        plot_path=np.asarray([str(plot_path)]),
    )
    print(f"budget={budget}")
    print(f"hash_mode={args.hash_mode}")
    print(f"toy_rounds={args.toy_rounds}")
    print(f"candidate_encoding={args.candidate_encoding}")
    print(f"input_dim={objective.input_dim}")
    print(f"generator_output_transform={args.generator_output_transform}")
    print(f"generator_output_temperature={args.generator_output_temperature}")
    print(f"generator_output_bias_init={args.generator_output_bias_init}")
    print(f"curiosity={args.curiosity}")
    print(f"curiosity_reference={args.curiosity_reference}")
    print(f"gfog_unique_candidates={len(objective.seen_candidates)}")
    print(f"random_unique_candidates={len(random_objective.seen_candidates)}")
    print(f"gfog_best={best:.12g}")
    print(f"random_best={random_best:.12g}")
    print(f"ratio_gfog_to_random={best / max(random_best, 1e-300):.4g}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
