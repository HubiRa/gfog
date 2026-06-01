"""Compare quantile and hybrid ranker objectives on cheap test functions."""

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
from gfog.opt import (  # noqa: E402
    HybridContextualUtilityRankerOpt,
    QuantileRankedDefaultOpt,
    components,
    make_torch_optimizer,
)
from gfog.opt.latents_sampler import LatentSamplerLambda  # noqa: E402


class DomainProjectedGenerator(nn.Module):
    """MLP generator projected into the test function domain."""

    def __init__(
        self,
        *,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        output_layernorm: bool,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            use_output_layernorm=output_layernorm,
        )
        self.register_buffer("lower", lower.reshape(1, -1).to(torch.float32))
        self.register_buffer("upper", upper.reshape(1, -1).to(torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        midpoint = 0.5 * (self.upper + self.lower)
        radius = 0.5 * (self.upper - self.lower)
        return midpoint + radius * torch.tanh(raw)


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


def build_optimizer(args: argparse.Namespace):
    test_function = make_function(args.function)
    if test_function.domain is None:
        raise ValueError(f"{args.function} has no domain")
    device = torch.device("cpu")
    value_levels = Levels(2) if args.function == "mishra_constrained" else Levels(1)
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
        output_layernorm=args.generator_output_layernorm,
    ).to(device)
    d = MLP(
        input_dim=test_function.input_dim,
        output_dim=1,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
        use_spectral_norm=args.discriminator_spectral_norm,
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
        fn=fn,
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=1,
        elite_sampling="random_top_k",
        elite_pool_size=args.buffer_multiplier * args.batch_size,
    )
    common_kwargs = {
        "ranker_list_size": args.ranker_list_size,
        "ranker_steps": 1,
        "ranker_sample_pool_size": args.ranker_sample_pool_size,
        "ranker_sample_mode": args.ranker_sample_mode,
    }
    if args.optimizer == "quantile":
        opt = QuantileRankedDefaultOpt(
            opt_components,
            ranker_weight=args.ranker_weight,
            ranker_target_curve=args.ranker_target_curve,
            ranker_tau=args.ranker_tau,
            ranker_target_scope=args.ranker_target_scope,
            **common_kwargs,
        )
    elif args.optimizer == "hybrid":
        opt = HybridContextualUtilityRankerOpt(
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
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return opt, test_function


def record_buffer_history(opt, iteration: int) -> dict[str, float]:
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
    ax.set_ylabel("objective, lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def random_search(
    test_function, *, budget: int, batch_size: int
) -> tuple[float, torch.Tensor]:
    best_value = float("inf")
    best_point = None
    if test_function.domain is None:
        raise ValueError("random_search requires a bounded domain")
    remaining = budget
    while remaining > 0:
        n = min(batch_size, remaining)
        remaining -= n
        x = test_function.domain.lower + (
            test_function.domain.upper - test_function.domain.lower
        ) * torch.rand(n, test_function.input_dim)
        result = test_function.f(x)
        if isinstance(result, list):
            objective = result[-1]
        else:
            objective = result
        local_best = int(torch.argmin(objective).item())
        local_value = float(objective[local_best].item())
        if local_value < best_value:
            best_value = local_value
            best_point = x[local_best].detach().clone()
    if best_point is None:
        raise RuntimeError("random_search evaluated no points")
    return best_value, best_point


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
    parser.add_argument(
        "--optimizer", choices=["quantile", "hybrid"], default="quantile"
    )
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.1)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--generator_output_layernorm", action="store_true")
    parser.add_argument("--discriminator_spectral_norm", action="store_true")
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
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
    parser.add_argument("--utility_target_scale", type=float, default=100.0)
    parser.add_argument(
        "--utility_loss", choices=["smooth_l1", "mse"], default="smooth_l1"
    )
    parser.add_argument("--utility_weight", type=float, default=0.1)
    parser.add_argument("--generator_utility_weight", type=float, default=0.1)
    parser.add_argument("--utility_clip", type=float, default=3.0)
    parser.add_argument("--history_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/testfunctions_ranker_objectives"),
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
        if args.history_interval > 0 and iteration % args.history_interval == 0:
            history.append(record_buffer_history(opt, iteration))

    best = opt.buffer.B.get_top_k(1).squeeze(0)
    best_values = opt.buffer.B.get_sorted_values()[0]
    best_value = float(best_values[-1])
    budget = (args.buffer_multiplier * args.batch_size) + args.n_iter * args.batch_size
    random_best, random_point = random_search(
        test_function,
        budget=budget,
        batch_size=args.batch_size,
    )
    distance = np.nan
    if test_function.known_minima is not None:
        distance = float(test_function.diff_from_minima(best.reshape(1, -1))[0].item())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"{args.function}_{args.optimizer}_iter{args.n_iter}_seed{args.seed}.npz"
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
        title=f"{args.function}: {args.optimizer}",
    )
    np.savez_compressed(
        output_path,
        function=np.asarray([args.function]),
        optimizer=np.asarray([args.optimizer]),
        best_point=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_values=np.asarray(best_values, dtype=np.float32),
        random_best=np.asarray([random_best], dtype=np.float32),
        random_point=random_point.detach().cpu().numpy(),
        budget=np.asarray([budget], dtype=np.int32),
        n_iter=np.asarray([args.n_iter], dtype=np.int32),
        batch_size=np.asarray([args.batch_size], dtype=np.int32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        ranker_list_size=np.asarray([args.ranker_list_size], dtype=np.int32),
        ranker_sample_pool_size=np.asarray(
            [args.ranker_sample_pool_size], dtype=np.int32
        ),
        ranker_target_curve=np.asarray([args.ranker_target_curve]),
        ranker_tau=np.asarray([args.ranker_tau], dtype=np.float32),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        utility_weight=np.asarray([args.utility_weight], dtype=np.float32),
        generator_utility_weight=np.asarray(
            [args.generator_utility_weight],
            dtype=np.float32,
        ),
        history=history_array,
        history_columns=np.asarray(
            ["iteration", "eval_count", "best", "mean", "median", "p10", "p90"]
        ),
        history_plot=np.asarray([str(history_plot_path)]),
        distance_to_minimum=np.asarray([distance], dtype=np.float32),
    )
    print(f"function={args.function}")
    print(f"optimizer={args.optimizer}")
    print(f"g_optimizer={args.g_optimizer}")
    print(f"d_optimizer={args.d_optimizer}")
    print(f"g_lr={args.g_lr}")
    print(f"d_lr={args.d_lr}")
    print(f"budget={budget}")
    print(f"best_value={best_value:.8f}")
    print(f"best_values={best_values}")
    print(f"best_point={best.tolist()}")
    print(f"random_best={random_best:.8f}")
    if not np.isnan(distance):
        print(f"distance_to_minimum={distance:.8f}")
    print(f"history_plot={history_plot_path}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
