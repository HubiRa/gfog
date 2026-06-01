"""TSP benchmark where GFog proposes route scores and f evaluates argsort routes."""

from __future__ import annotations

import argparse
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
from gfog.models import MLP, OutputNormalizer  # noqa: E402
from gfog.opt import (  # noqa: E402
    HybridContextualUtilityRankerOpt,
    LSGANOpt,
    QuantileRankedDefaultOpt,
    components,
)
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
    """Small experimental Muon optimizer for matrix-heavy networks."""

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


class ResidualRoutePriorScores(nn.Module):
    """Add a fixed route-rank prior to generated residual route scores."""

    def __init__(
        self,
        generator: nn.Module,
        base_scores: torch.Tensor,
        alpha: float,
    ) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError(f"route_prior_alpha must be non-negative, got {alpha}")
        if base_scores.ndim != 1:
            raise ValueError(
                f"base_scores must be 1D, got shape={tuple(base_scores.shape)}"
            )
        self.generator = generator
        self.alpha = alpha
        self.register_buffer("base_scores", base_scores.reshape(1, -1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        residual = self.generator(z)
        base = self.base_scores.to(device=residual.device, dtype=residual.dtype)
        return base + self.alpha * residual


class SetTransformerGenerator(nn.Module):
    """Batch-aware set transformer generator that emits route score vectors."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        model_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"set generator model_dim must be divisible by heads, got {model_dim} and {heads}"
            )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"SetTransformerGenerator expects 2D input, got {z.shape}")
        tokens = self.token_proj(z).unsqueeze(0)
        encoded = self.encoder(tokens).squeeze(0)
        return self.output_head(encoded)


class SetTransformerDiscriminator(nn.Module):
    """Permutation-equivariant listwise discriminator over route score vectors."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"set discriminator model_dim must be divisible by heads, got {model_dim} and {heads}"
            )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.score_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"SetTransformerDiscriminator expects 2D input, got {x.shape}"
            )
        tokens = self.input_proj(x).unsqueeze(0)
        encoded = self.encoder(tokens).squeeze(0)
        return self.score_head(encoded)


def make_cities(n_cities: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2), dtype=np.float64)


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    delta = cities[:, None, :] - cities[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def route_lengths_from_orders(orders: np.ndarray, dist: np.ndarray) -> np.ndarray:
    starts = orders
    ends = np.roll(orders, shift=-1, axis=1)
    return dist[starts, ends].sum(axis=1)


class TSPArgsortObjective:
    """Evaluate score vectors by sorting them into TSP routes."""

    def __init__(self, cities: np.ndarray, two_opt_passes: int = 0) -> None:
        self.cities = np.asarray(cities, dtype=np.float64)
        if self.cities.ndim != 2 or self.cities.shape[1] != 2:
            raise ValueError(f"cities must have shape (n, 2), got {self.cities.shape}")
        if two_opt_passes < 0:
            raise ValueError(
                f"two_opt_passes must be non-negative, got {two_opt_passes}"
            )
        self.dist = distance_matrix(self.cities)
        self.two_opt_passes = two_opt_passes
        self.evaluation_count = 0

    @property
    def n_cities(self) -> int:
        return int(self.cities.shape[0])

    def route_from_scores(
        self,
        scores: torch.Tensor | np.ndarray,
        *,
        apply_two_opt: bool | None = None,
    ) -> np.ndarray:
        if isinstance(scores, torch.Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)
        routes = np.argsort(scores_np, axis=1, kind="stable")
        should_optimize = (
            self.two_opt_passes > 0 if apply_two_opt is None else apply_two_opt
        )
        if not should_optimize:
            return routes
        return np.stack(
            [
                two_opt(route, self.dist, max_passes=self.two_opt_passes)
                for route in routes
            ],
            axis=0,
        )

    def __call__(self, scores: torch.Tensor | np.ndarray) -> list[float]:
        orders = self.route_from_scores(scores)
        lengths = route_lengths_from_orders(orders, self.dist)
        self.evaluation_count += int(orders.shape[0])
        return lengths.astype(np.float64).tolist()


def random_permutation_baseline(
    objective: TSPArgsortObjective,
    *,
    budget: int,
    seed: int,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    best = float("inf")
    curve: list[float] = []
    for _ in range(budget):
        route = rng.permutation(objective.n_cities).reshape(1, -1)
        length = float(route_lengths_from_orders(route, objective.dist)[0])
        best = min(best, length)
        curve.append(best)
    return best, curve


def nearest_neighbor_route(
    objective: TSPArgsortObjective, start: int = 0
) -> np.ndarray:
    n = objective.n_cities
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        current = route[-1]
        nxt = min(unvisited, key=lambda idx: objective.dist[current, idx])
        route.append(nxt)
        unvisited.remove(nxt)
    return np.asarray(route, dtype=np.int64)


def route_to_rank_scores(route: np.ndarray) -> np.ndarray:
    """Convert a route order into per-city rank scores consumed by argsort."""
    ranks = np.empty_like(route, dtype=np.float64)
    ranks[route.astype(np.int64)] = np.arange(route.shape[0], dtype=np.float64)
    return ranks


def hilbert_index(x: int, y: int, bits: int) -> int:
    """Return Hilbert curve index for integer coordinates in [0, 2**bits)."""
    index = 0
    n = 1 << bits
    scale = n >> 1
    while scale > 0:
        rx = 1 if (x & scale) else 0
        ry = 1 if (y & scale) else 0
        index += scale * scale * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        scale >>= 1
    return index


def hilbert_route(cities: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        raise ValueError(f"route_prior_hilbert_bits must be positive, got {bits}")
    max_coord = (1 << bits) - 1
    clipped = np.clip(cities, 0.0, 1.0)
    coords = np.rint(clipped * max_coord).astype(np.int64)
    indices = np.asarray(
        [hilbert_index(int(x), int(y), bits) for x, y in coords],
        dtype=np.int64,
    )
    return np.lexsort((np.arange(cities.shape[0]), indices))


def route_prior_scores(
    objective: TSPArgsortObjective,
    prior: str,
    *,
    hilbert_bits: int,
) -> np.ndarray:
    """Build per-city base rank scores for a geometric route prior."""
    if prior == "none":
        return np.zeros(objective.n_cities, dtype=np.float64)
    if prior == "x":
        return route_to_rank_scores(np.argsort(objective.cities[:, 0], kind="stable"))
    if prior == "y":
        return route_to_rank_scores(np.argsort(objective.cities[:, 1], kind="stable"))
    if prior == "nearest_neighbor":
        return route_to_rank_scores(nearest_neighbor_route(objective, start=0))
    if prior == "hilbert":
        return route_to_rank_scores(hilbert_route(objective.cities, bits=hilbert_bits))
    raise ValueError(f"Unknown route_prior: {prior}")


def two_opt(route: np.ndarray, dist: np.ndarray, max_passes: int = 50) -> np.ndarray:
    route = route.copy()
    n = int(route.shape[0])
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            a = route[i]
            b = route[(i + 1) % n]
            for j in range(i + 2, n):
                c = route[j]
                d = route[(j + 1) % n]
                old = dist[a, b] + dist[c, d]
                new = dist[a, c] + dist[b, d]
                if new + 1e-12 < old:
                    route[i + 1 : j + 1] = route[i + 1 : j + 1][::-1]
                    improved = True
        if not improved:
            break
    return route


def route_length(route: np.ndarray, dist: np.ndarray) -> float:
    return float(route_lengths_from_orders(route.reshape(1, -1), dist)[0])


def two_opt_random_baseline(
    objective: TSPArgsortObjective,
    *,
    starts: int,
    seed: int,
    max_passes: int,
) -> float:
    rng = np.random.default_rng(seed)
    best = float("inf")
    for _ in range(starts):
        route = rng.permutation(objective.n_cities)
        route = two_opt(route, objective.dist, max_passes=max_passes)
        best = min(best, route_length(route, objective.dist))
    return best


def build_optimizer(args: argparse.Namespace, objective: TSPArgsortObjective):
    device = torch.device("cpu")
    if args.generator_type == "mlp":
        g_base = MLP(
            input_dim=args.latent_dim,
            output_dim=args.n_cities,
            hidden_dims=args.generator_hidden_dims,
        ).to(device)
    elif args.generator_type == "set_transformer":
        g_base = SetTransformerGenerator(
            latent_dim=args.latent_dim,
            output_dim=args.n_cities,
            model_dim=args.set_generator_dim,
            depth=args.set_generator_depth,
            heads=args.set_generator_heads,
            mlp_ratio=args.set_generator_mlp_ratio,
            dropout=args.set_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")
    g = OutputNormalizer(g_base, args.generator_output_norm).to(device)
    if args.route_prior != "none":
        base_scores = torch.as_tensor(
            route_prior_scores(
                objective,
                args.route_prior,
                hilbert_bits=args.route_prior_hilbert_bits,
            ),
            dtype=torch.float32,
        )
        g = ResidualRoutePriorScores(
            g,
            base_scores=base_scores,
            alpha=args.route_prior_alpha,
        ).to(device)
    if args.discriminator_type == "mlp":
        d = MLP(
            input_dim=args.n_cities,
            output_dim=1,
            hidden_dims=args.discriminator_hidden_dims,
            use_spectral_norm=args.discriminator_spectral_norm,
        ).to(device)
    elif args.discriminator_type == "set_transformer":
        d = SetTransformerDiscriminator(
            input_dim=args.n_cities,
            model_dim=args.set_discriminator_dim,
            depth=args.set_discriminator_depth,
            heads=args.set_discriminator_heads,
            mlp_ratio=args.set_discriminator_mlp_ratio,
            dropout=args.set_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown discriminator_type: {args.discriminator_type}")
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
            input_dim=args.n_cities,
            device=device,
            dtype=torch.float32,
        ),
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=1,
    )
    common_kwargs = {
        "ranker_list_size": args.ranker_list_size,
        "ranker_steps": 1,
        "ranker_sample_pool_size": args.ranker_sample_pool_size,
        "ranker_sample_mode": args.ranker_sample_mode,
    }
    if args.optimizer == "quantile":
        return QuantileRankedDefaultOpt(
            opt_components,
            ranker_weight=args.ranker_weight,
            ranker_target_curve=args.ranker_target_curve,
            ranker_tau=args.ranker_tau,
            ranker_target_scope=args.ranker_target_scope,
            **common_kwargs,
        )
    if args.optimizer == "hybrid":
        return HybridContextualUtilityRankerOpt(
            opt_components,
            d_score_center_weight=args.d_score_center_weight,
            d_score_scale_weight=args.d_score_scale_weight,
            d_score_target_std=args.d_score_target_std,
            utility_target_scale=args.utility_target_scale,
            utility_loss=args.utility_loss,
            utility_weight=args.utility_weight,
            generator_utility_weight=args.generator_utility_weight,
            utility_clip=args.utility_clip,
            **common_kwargs,
        )
    if args.optimizer == "lsgan":
        return LSGANOpt(opt_components)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def record_buffer_history(opt, iteration: int) -> dict[str, float]:
    values = opt.buffer.B.get_sorted_values()
    lengths = np.asarray([row[-1] for row in values], dtype=np.float64)
    return {
        "iteration": float(iteration),
        "eval_count": float(
            opt.buffer.B.buffer_size + iteration * opt.components.batch_size
        ),
        "best": float(lengths[0]),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p10": float(np.percentile(lengths, 10)),
        "p90": float(np.percentile(lengths, 90)),
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
    ax.plot(evals, best, label="best buffer tour", linewidth=2)
    ax.plot(evals, mean, label="mean buffer tour", linewidth=1.5)
    ax.plot(evals, median, label="median buffer tour", linewidth=1.5)
    ax.fill_between(evals, p10, p90, alpha=0.18, label="p10-p90 buffer range")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("tour length, lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def format_dims(dims: list[int]) -> str:
    return "x".join(str(dim) for dim in dims) if dims else "linear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cities", type=int, default=50)
    parser.add_argument("--city_seed", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=["quantile", "hybrid", "lsgan"],
        default="quantile",
    )
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument(
        "--generator_type", choices=["mlp", "set_transformer"], default="mlp"
    )
    parser.add_argument("--generator_hidden_dims", type=parse_ints, default=[128, 128])
    parser.add_argument(
        "--discriminator_type",
        choices=["mlp", "set_transformer"],
        default="mlp",
    )
    parser.add_argument(
        "--discriminator_hidden_dims", type=parse_ints, default=[128, 128]
    )
    parser.add_argument("--set_generator_dim", type=int, default=256)
    parser.add_argument("--set_generator_depth", type=int, default=2)
    parser.add_argument("--set_generator_heads", type=int, default=4)
    parser.add_argument("--set_generator_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_discriminator_dim", type=int, default=256)
    parser.add_argument("--set_discriminator_depth", type=int, default=2)
    parser.add_argument("--set_discriminator_heads", type=int, default=4)
    parser.add_argument("--set_discriminator_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_dropout", type=float, default=0.0)
    parser.add_argument(
        "--generator_output_norm",
        choices=["none", "l2", "centered_l2", "layernorm"],
        default="centered_l2",
    )
    parser.add_argument(
        "--route_prior",
        choices=["none", "x", "y", "nearest_neighbor", "hilbert"],
        default="none",
        help="Add fixed per-city base rank scores before argsort.",
    )
    parser.add_argument(
        "--route_prior_alpha",
        type=float,
        default=1.0,
        help="Scale of learned residual scores added to the fixed route prior.",
    )
    parser.add_argument(
        "--route_prior_hilbert_bits",
        type=int,
        default=10,
        help="Grid resolution bits used by the Hilbert route prior.",
    )
    parser.add_argument("--discriminator_spectral_norm", action="store_true")
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.1)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--curiosity", type=float, default=0.0)
    parser.add_argument("--curiosity_t", type=float, default=2.0)
    parser.add_argument(
        "--curiosity_reference",
        choices=["batch", "buffer"],
        default="batch",
    )
    parser.add_argument("--ranker_list_size", type=int, default=128)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=512)
    parser.add_argument(
        "--ranker_sample_mode",
        choices=["random_top_pool", "top_k"],
        default="random_top_pool",
    )
    parser.add_argument("--ranker_weight", type=float, default=1.0)
    parser.add_argument(
        "--ranker_target_curve", choices=["linear", "exp"], default="exp"
    )
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument(
        "--ranker_target_scope", choices=["local", "global"], default="local"
    )
    parser.add_argument("--d_score_center_weight", type=float, default=0.01)
    parser.add_argument("--d_score_scale_weight", type=float, default=0.01)
    parser.add_argument("--d_score_target_std", type=float, default=1.0)
    parser.add_argument("--utility_target_scale", type=float, default=10.0)
    parser.add_argument(
        "--utility_loss", choices=["smooth_l1", "mse"], default="smooth_l1"
    )
    parser.add_argument("--utility_weight", type=float, default=0.1)
    parser.add_argument("--generator_utility_weight", type=float, default=0.1)
    parser.add_argument("--utility_clip", type=float, default=3.0)
    parser.add_argument(
        "--objective_two_opt_passes",
        type=int,
        default=0,
        help="Apply this many 2-opt passes inside f after argsort before scoring.",
    )
    parser.add_argument("--two_opt_starts", type=int, default=64)
    parser.add_argument("--two_opt_passes", type=int, default=50)
    parser.add_argument("--history_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("results/tsp_argsort"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_cities < 3:
        raise ValueError(f"n_cities must be at least 3, got {args.n_cities}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cities = make_cities(args.n_cities, args.city_seed)
    objective = TSPArgsortObjective(
        cities, two_opt_passes=args.objective_two_opt_passes
    )
    opt = build_optimizer(args, objective)
    history = [record_buffer_history(opt, 0)]
    for iteration in range(1, args.n_iter + 1):
        opt.step()
        if args.history_interval > 0 and iteration % args.history_interval == 0:
            history.append(record_buffer_history(opt, iteration))

    best_scores = opt.buffer.B.get_top_k(1).detach().cpu()
    best_raw_route = objective.route_from_scores(best_scores, apply_two_opt=False)[0]
    best_route = objective.route_from_scores(best_scores, apply_two_opt=True)[0]
    best_length = float(opt.buffer.B.get_value(0))
    best_raw_length = route_length(best_raw_route, objective.dist)
    budget = args.buffer_multiplier * args.batch_size + args.n_iter * args.batch_size
    random_best, random_curve = random_permutation_baseline(
        objective,
        budget=budget,
        seed=args.seed + 101,
    )
    nn_route = nearest_neighbor_route(objective, start=0)
    nn_length = route_length(nn_route, objective.dist)
    two_opt_nn = two_opt(nn_route, objective.dist, max_passes=args.two_opt_passes)
    two_opt_nn_length = route_length(two_opt_nn, objective.dist)
    two_opt_random_length = two_opt_random_baseline(
        objective,
        starts=args.two_opt_starts,
        seed=args.seed + 202,
        max_passes=args.two_opt_passes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"tsp_city{args.city_seed}_n{args.n_cities}_{args.optimizer}_"
        f"gtype{args.generator_type}_dtype{args.discriminator_type}_"
        f"gh{format_dims(args.generator_hidden_dims)}_"
        f"dh{format_dims(args.discriminator_hidden_dims)}_"
        f"sg{args.set_generator_dim}x{args.set_generator_depth}h{args.set_generator_heads}_"
        f"sd{args.set_discriminator_dim}x{args.set_discriminator_depth}h{args.set_discriminator_heads}_"
        f"b{args.batch_size}_bm{args.buffer_multiplier}_"
        f"rank{args.ranker_list_size}_pool{args.ranker_sample_pool_size}_"
        f"tau{args.ranker_tau:g}_norm{args.generator_output_norm}_"
        f"prior{args.route_prior}_pa{args.route_prior_alpha:g}_"
        f"f2opt{args.objective_two_opt_passes}_"
        f"gopt{args.g_optimizer}_dopt{args.d_optimizer}_"
        f"glr{args.g_lr:g}_dlr{args.d_lr:g}_curio{args.curiosity:g}"
        f"_{args.curiosity_reference}_ct{args.curiosity_t:g}_"
        f"iter{args.n_iter}_seed{args.seed}.npz"
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
        history, history_plot_path, title=f"TSP N={args.n_cities}: {args.optimizer}"
    )
    np.savez_compressed(
        output_path,
        optimizer=np.asarray([args.optimizer]),
        n_cities=np.asarray([args.n_cities], dtype=np.int32),
        city_seed=np.asarray([args.city_seed], dtype=np.int32),
        seed=np.asarray([args.seed], dtype=np.int32),
        cities=cities.astype(np.float32),
        best_raw_route=best_raw_route.astype(np.int32),
        best_route=best_route.astype(np.int32),
        best_raw_length=np.asarray([best_raw_length], dtype=np.float32),
        best_length=np.asarray([best_length], dtype=np.float32),
        random_best=np.asarray([random_best], dtype=np.float32),
        random_curve=np.asarray(random_curve, dtype=np.float32),
        nearest_neighbor_length=np.asarray([nn_length], dtype=np.float32),
        two_opt_nearest_neighbor_length=np.asarray(
            [two_opt_nn_length], dtype=np.float32
        ),
        two_opt_random_length=np.asarray([two_opt_random_length], dtype=np.float32),
        budget=np.asarray([budget], dtype=np.int32),
        n_iter=np.asarray([args.n_iter], dtype=np.int32),
        batch_size=np.asarray([args.batch_size], dtype=np.int32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        latent_dim=np.asarray([args.latent_dim], dtype=np.int32),
        generator_type=np.asarray([args.generator_type]),
        discriminator_type=np.asarray([args.discriminator_type]),
        generator_hidden_dims=np.asarray(args.generator_hidden_dims, dtype=np.int32),
        discriminator_hidden_dims=np.asarray(
            args.discriminator_hidden_dims, dtype=np.int32
        ),
        set_generator_dim=np.asarray([args.set_generator_dim], dtype=np.int32),
        set_generator_depth=np.asarray([args.set_generator_depth], dtype=np.int32),
        set_generator_heads=np.asarray([args.set_generator_heads], dtype=np.int32),
        set_generator_mlp_ratio=np.asarray(
            [args.set_generator_mlp_ratio], dtype=np.int32
        ),
        set_discriminator_dim=np.asarray([args.set_discriminator_dim], dtype=np.int32),
        set_discriminator_depth=np.asarray(
            [args.set_discriminator_depth], dtype=np.int32
        ),
        set_discriminator_heads=np.asarray(
            [args.set_discriminator_heads], dtype=np.int32
        ),
        set_discriminator_mlp_ratio=np.asarray(
            [args.set_discriminator_mlp_ratio], dtype=np.int32
        ),
        set_dropout=np.asarray([args.set_dropout], dtype=np.float32),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        ranker_list_size=np.asarray([args.ranker_list_size], dtype=np.int32),
        ranker_sample_pool_size=np.asarray(
            [args.ranker_sample_pool_size], dtype=np.int32
        ),
        ranker_sample_mode=np.asarray([args.ranker_sample_mode]),
        ranker_tau=np.asarray([args.ranker_tau], dtype=np.float32),
        ranker_target_curve=np.asarray([args.ranker_target_curve]),
        ranker_target_scope=np.asarray([args.ranker_target_scope]),
        route_prior=np.asarray([args.route_prior]),
        route_prior_alpha=np.asarray([args.route_prior_alpha], dtype=np.float32),
        route_prior_hilbert_bits=np.asarray(
            [args.route_prior_hilbert_bits], dtype=np.int32
        ),
        objective_two_opt_passes=np.asarray(
            [args.objective_two_opt_passes], dtype=np.int32
        ),
        generator_output_norm=np.asarray([args.generator_output_norm]),
        discriminator_spectral_norm=np.asarray([args.discriminator_spectral_norm]),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        curiosity_t=np.asarray([args.curiosity_t], dtype=np.float32),
        curiosity_reference=np.asarray([args.curiosity_reference]),
        history=history_array,
        history_columns=np.asarray(
            ["iteration", "eval_count", "best", "mean", "median", "p10", "p90"]
        ),
        history_plot=np.asarray([str(history_plot_path)]),
    )
    print(f"n_cities={args.n_cities}")
    print(f"optimizer={args.optimizer}")
    print(f"budget={budget}")
    print(f"best_length={best_length:.6f}")
    print(f"best_raw_length={best_raw_length:.6f}")
    print(f"random_best={random_best:.6f}")
    print(f"nearest_neighbor={nn_length:.6f}")
    print(f"two_opt_nearest_neighbor={two_opt_nn_length:.6f}")
    print(f"two_opt_random={two_opt_random_length:.6f}")
    print(f"history_plot={history_plot_path}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
