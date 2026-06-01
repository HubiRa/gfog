"""Compare ranked GFog variants on simple test functions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import BCEWithLogitsLoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from functions import (  # noqa: E402
    AckleyFunction,
    BealeFunction,
    GoldsteinPriceFunction,
    HimmelblauFunction,
    MishrasBirdFunctionConstraint,
    Rosenbrock2DFunction,
    ThreeHumpCamelFunction,
)
from gfog.buffer import Buffer, Levels  # noqa: E402
from gfog.models import MLP  # noqa: E402
from gfog.opt import BaseOpt, components  # noqa: E402
from gfog.opt.latents_sampler import LatentSamplerLambda  # noqa: E402


class DomainProjectedGenerator(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim, hidden_dim],
        )
        self.register_buffer("lower", lower.reshape(1, -1).to(torch.float32))
        self.register_buffer("upper", upper.reshape(1, -1).to(torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        midpoint = 0.5 * (self.upper + self.lower)
        radius = 0.5 * (self.upper - self.lower)
        return midpoint + radius * torch.tanh(raw)


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


def rank_by_values(candidates: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.detach().reshape(-1), descending=False)
    return candidates[order]


class RankedOpt(BaseOpt):
    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int,
        ranker_sample_pool_size: int,
        ranker_tau: float,
        mixed_rank_update: bool,
        use_fake_loss: bool,
    ) -> None:
        super().__init__(opt_components)
        self.ranker_list_size = ranker_list_size
        self.ranker_sample_pool_size = ranker_sample_pool_size
        self.ranker_tau = ranker_tau
        self.mixed_rank_update = mixed_rank_update
        self.use_fake_loss = use_fake_loss

    def _ranked_buffer_subset(self) -> torch.Tensor:
        current_len = len(self.buffer.B)
        pool_size = min(
            current_len, max(self.ranker_list_size, self.ranker_sample_pool_size)
        )
        k = min(self.ranker_list_size, pool_size)
        if k == pool_size:
            ranked = self.buffer.B.get_top_k(k)
        else:
            positions = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.B.get(int(pos)) for pos in positions])
        return ranked.to(self.gan.device, self.gan.dtype)

    def _ranked_buffer_subset_with_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranked = self.buffer.B.get_top_k(min(self.ranker_list_size, len(self.buffer.B)))
        values = torch.as_tensor(
            [row[-1] for row in self.buffer.B.get_sorted_values()[: ranked.shape[0]]],
            dtype=self.gan.dtype,
            device=self.gan.device,
        )
        return ranked.to(self.gan.device, self.gan.dtype), values

    def _train_discriminator_on_ranked(self, ranked: torch.Tensor) -> None:
        self.gan.optimizerD.zero_grad()
        scores = self.gan.D(ranked)
        targets = rank_targets(
            scores.numel(),
            device=scores.device,
            dtype=scores.dtype,
            tau=self.ranker_tau,
        ).reshape_as(scores)
        loss = F.mse_loss(scores, targets)
        loss.backward()
        self.gan.optimizerD.step()

    def _train_discriminator_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self._ranked_buffer_subset()
        scores = self.gan.D(ranked)
        targets = rank_targets(
            scores.numel(),
            device=scores.device,
            dtype=scores.dtype,
            tau=self.ranker_tau,
        ).reshape_as(scores)
        loss = F.mse_loss(scores, targets)
        if self.use_fake_loss:
            with torch.no_grad():
                z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
                fake = self.gan.G(z)
            fake_scores = self.gan.D(fake.detach())
            loss = loss + F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_step()
        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = F.mse_loss(scores, torch.ones_like(scores))
        loss.backward()
        self.gan.optimizerG.step()
        return proposals

    def evaluate(self, proposals: torch.Tensor) -> None:
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))

    def step(self) -> None:
        if not self.mixed_rank_update:
            return super().step()

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            evaluated = self.gan.G(z)
            values = self.fn.f(evaluated.detach().to(self.fn.device, self.fn.dtype))

        buffer_batch, buffer_values = self._ranked_buffer_subset_with_values()
        mixed = torch.cat([buffer_batch, evaluated.detach()], dim=0)
        mixed_values = torch.cat(
            [
                buffer_values,
                torch.as_tensor(
                    values, dtype=self.gan.dtype, device=self.gan.device
                ).reshape(-1),
            ],
            dim=0,
        )
        ranked = rank_by_values(mixed, mixed_values)
        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_on_ranked(ranked)

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = F.mse_loss(scores, torch.ones_like(scores))
        loss.backward()
        self.gan.optimizerG.step()

        self.buffer.B.insert_many(values=list(values), tensors=list(evaluated.detach()))


def make_function(name: str):
    if name == "ackley":
        return AckleyFunction()
    if name == "beale":
        return BealeFunction()
    if name == "goldstein_price":
        return GoldsteinPriceFunction()
    if name == "himmelblau":
        return HimmelblauFunction()
    if name == "mishra":
        return MishrasBirdFunctionConstraint(return_constraints=False)
    if name == "mishra_constrained":
        return MishrasBirdFunctionConstraint(return_constraints=True)
    if name == "rosenbrock":
        return Rosenbrock2DFunction()
    if name == "three_hump_camel":
        return ThreeHumpCamelFunction()
    raise ValueError(f"Unknown function: {name}")


def random_search(
    test_function, *, budget: int, batch_size: int
) -> tuple[float, torch.Tensor]:
    best_value = float("inf")
    best_point = None
    assert test_function.domain is not None
    for _ in range(int(np.ceil(budget / batch_size))):
        n = min(batch_size, budget)
        budget -= n
        x = test_function.domain.lower + (
            test_function.domain.upper - test_function.domain.lower
        ) * torch.rand(n, test_function.input_dim)
        values = torch.as_tensor(test_function(x), dtype=torch.float32).reshape(-1)
        idx = int(torch.argmin(values).item())
        if float(values[idx].item()) < best_value:
            best_value = float(values[idx].item())
            best_point = x[idx].detach().clone()
    assert best_point is not None
    return best_value, best_point


def build_optimizer(args: argparse.Namespace):
    test_function = make_function(args.function)
    if args.function == "mishra_constrained":
        value_levels: Levels | int = Levels(["constraint", "fx"])
    else:
        value_levels = 1
    assert test_function.domain is not None
    device = torch.device("cpu")
    fn = components.Fn(
        f=test_function,
        input_dim=test_function.input_dim,
        device=device,
        dtype=torch.float32,
    )
    g = DomainProjectedGenerator(
        latent_dim=args.latent_dim,
        output_dim=test_function.input_dim,
        hidden_dim=args.hidden_dim,
        lower=test_function.domain.lower,
        upper=test_function.domain.upper,
    ).to(device)
    d = MLP(
        input_dim=test_function.input_dim,
        output_dim=1,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    ).to(device)
    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=value_levels,
        )
    )
    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=None,
        latent_dim=args.latent_dim,
        optimizerG=torch.optim.Adam(g.parameters(), lr=args.g_lr),
        optimizerD=torch.optim.Adam(d.parameters(), lr=args.d_lr),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d),
            b=args.batch_size,
            d=args.latent_dim,
        ),
        device=device,
        dtype=torch.float32,
    )
    opt_components = components.OptComponents(
        fn=fn,
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=args.discriminator_steps,
        elite_sampling="random_top_k",
        elite_pool_size=args.buffer_multiplier * args.batch_size,
    )
    opt = RankedOpt(
        opt_components,
        ranker_list_size=args.ranker_list_size,
        ranker_sample_pool_size=args.ranker_sample_pool_size,
        ranker_tau=args.ranker_tau,
        mixed_rank_update=args.mixed_rank_update,
        use_fake_loss=not args.no_fake_loss,
    )
    return opt, test_function


def record_buffer_history(opt: RankedOpt, iteration: int) -> dict[str, float]:
    values = opt.buffer.B.get_sorted_values()
    last_level = np.asarray([row[-1] for row in values], dtype=np.float64)
    return {
        "iteration": float(iteration),
        "eval_count": float(
            opt.buffer.B.buffer_size + iteration * opt.components.batch_size
        ),
        "best": float(last_level[0]),
        "mean": float(np.mean(last_level)),
        "median": float(np.median(last_level)),
        "p10": float(np.percentile(last_level, 10)),
        "p90": float(np.percentile(last_level, 90)),
    }


def plot_history(
    history: list[dict[str, float]], output_path: Path, title: str
) -> None:
    if not history:
        return
    evals = np.asarray([row["eval_count"] for row in history])
    best = np.asarray([row["best"] for row in history])
    mean = np.asarray([row["mean"] for row in history])
    median = np.asarray([row["median"] for row in history])
    p10 = np.asarray([row["p10"] for row in history])
    p90 = np.asarray([row["p90"] for row in history])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(evals, best, label="best buffer score", linewidth=2)
    ax.plot(evals, mean, label="mean buffer score", linewidth=1.5)
    ax.plot(evals, median, label="median buffer score", linewidth=1.5)
    ax.fill_between(evals, p10, p90, alpha=0.18, label="p10-p90 buffer range")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("buffer objective, lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--function",
        choices=[
            "ackley",
            "beale",
            "goldstein_price",
            "himmelblau",
            "mishra",
            "mishra_constrained",
            "rosenbrock",
            "three_hump_camel",
        ],
        default="mishra",
    )
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=3e-3)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--mixed_rank_update", action="store_true")
    parser.add_argument("--no_fake_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/testfunctions_ranked")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    opt, test_function = build_optimizer(args)
    history = [record_buffer_history(opt, 0)]
    for iteration in range(1, args.n_iter + 1):
        opt.step()
        history.append(record_buffer_history(opt, iteration))

    best = opt.buffer.B.get_top_k(1).squeeze(0)
    best_value = float(opt.buffer.B.get_value(0, level=-1))
    best_values = opt.buffer.B.get_sorted_values()[0]
    budget = (args.buffer_multiplier * args.batch_size) + args.n_iter * args.batch_size
    random_best, random_point = random_search(
        test_function,
        budget=budget,
        batch_size=args.batch_size,
    )
    distance = None
    if test_function.known_minima is not None:
        distance = float(test_function.diff_from_minima(best.reshape(1, -1))[0].item())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"{args.function}_mixed{int(args.mixed_rank_update)}_seed_{args.seed}.npz"
    )
    history_array = np.asarray(
        [
            [
                row["iteration"],
                row["eval_count"],
                row["best"],
                row["mean"],
                row["median"],
                row["p10"],
                row["p90"],
            ]
            for row in history
        ],
        dtype=np.float32,
    )
    history_plot_path = output_path.with_suffix(".png")
    plot_history(
        history,
        history_plot_path,
        title=f"{args.function}: mixed={args.mixed_rank_update}, fake={not args.no_fake_loss}",
    )
    np.savez_compressed(
        output_path,
        function=np.asarray([args.function]),
        mixed_rank_update=np.asarray([args.mixed_rank_update]),
        use_fake_loss=np.asarray([not args.no_fake_loss]),
        best_point=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_values=np.asarray(best_values, dtype=np.float32),
        random_best=np.asarray([random_best], dtype=np.float32),
        random_point=random_point.detach().cpu().numpy(),
        budget=np.asarray([budget], dtype=np.int32),
        history=history_array,
        history_columns=np.asarray(
            ["iteration", "eval_count", "best", "mean", "median", "p10", "p90"]
        ),
        history_plot=np.asarray([str(history_plot_path)]),
        distance_to_minimum=np.asarray(
            [np.nan if distance is None else distance], dtype=np.float32
        ),
    )
    print(f"function={args.function}")
    print(f"mixed_rank_update={args.mixed_rank_update}")
    print(f"use_fake_loss={not args.no_fake_loss}")
    print(f"budget={budget}")
    print(f"best_value={best_value:.8f}")
    print(f"best_values={best_values}")
    print(f"best_point={best.tolist()}")
    print(f"random_best={random_best:.8f}")
    if distance is not None:
        print(f"distance_to_minimum={distance:.8f}")
    print(f"history_plot={history_plot_path}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
