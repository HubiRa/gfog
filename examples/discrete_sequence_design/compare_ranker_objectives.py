"""Compare quantile and hybrid ranker objectives on discrete motif design."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from motif_task import (  # noqa: E402
    ConvSequenceDiscriminator,
    ConvSequenceGenerator,
    MLPSequenceDiscriminator,
    MLPSequenceGenerator,
    MotifObjective,
    MotifProblem,
    make_optimizer,
    random_search,
)
from gfog.buffer import Buffer  # noqa: E402
from gfog.models import OutputNormalizer  # noqa: E402
from gfog.opt import (  # noqa: E402
    HybridContextualUtilityRankerOpt,
    QuantileRankedDefaultOpt,
    components,
)
from gfog.opt.latents_sampler import LatentSamplerLambda  # noqa: E402


def build_optimizer(
    args: argparse.Namespace,
) -> tuple[
    QuantileRankedDefaultOpt | HybridContextualUtilityRankerOpt,
    MotifObjective,
    MotifProblem,
]:
    problem = MotifProblem(
        length=args.length,
        alphabet_size=args.alphabet_size,
        motif_length=args.motif_length,
        n_motifs=args.n_motifs,
        seed=args.seed,
        position_mode=args.position_mode,
    )
    objective = MotifObjective(problem)
    input_dim = args.length * args.alphabet_size
    device = torch.device("cpu")
    if args.model_type == "conv":
        generator = ConvSequenceGenerator(
            latent_dim=args.latent_dim,
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
        )
        discriminator = ConvSequenceDiscriminator(
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
        )
    else:
        generator = MLPSequenceGenerator(
            latent_dim=args.latent_dim,
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
        )
        discriminator = MLPSequenceDiscriminator(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
        )
    generator = OutputNormalizer(generator, args.generator_output_norm).to(device)
    discriminator = discriminator.to(device)
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
    )
    gan = components.GAN(
        G=generator,
        D=discriminator,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=None,
        latent_dim=args.latent_dim,
        optimizerG=make_optimizer(
            args.g_optimizer,
            generator.parameters(),
            lr=args.g_lr,
            momentum=args.optimizer_momentum,
        ),
        optimizerD=make_optimizer(
            args.d_optimizer,
            discriminator.parameters(),
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
            input_dim=input_dim,
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
    return opt, objective, problem


def record_buffer_history(
    opt, iteration: int, evaluation_count: int
) -> dict[str, float]:
    values = opt.buffer.B.get_sorted_values()
    objective = np.asarray([row[-1] for row in values], dtype=np.float64)
    scores = -objective
    return {
        "iteration": float(iteration),
        "eval_count": float(evaluation_count),
        "best_score": float(scores[0]),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "p10_score": float(np.percentile(scores, 10)),
        "p90_score": float(np.percentile(scores, 90)),
    }


def plot_history(
    history: list[dict[str, float]], output_path: Path, title: str
) -> None:
    if not history:
        return
    evals = np.asarray([row["eval_count"] for row in history])
    best = np.asarray([row["best_score"] for row in history])
    mean = np.asarray([row["mean_score"] for row in history])
    median = np.asarray([row["median_score"] for row in history])
    p10 = np.asarray([row["p10_score"] for row in history])
    p90 = np.asarray([row["p90_score"] for row in history])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(evals, best, label="best buffer score", linewidth=2)
    ax.plot(evals, mean, label="mean buffer score", linewidth=1.5)
    ax.plot(evals, median, label="median buffer score", linewidth=1.5)
    ax.fill_between(evals, p10, p90, alpha=0.18, label="p10-p90 buffer range")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("motif score, higher is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--alphabet_size", type=int, default=8)
    parser.add_argument("--motif_length", type=int, default=8)
    parser.add_argument("--n_motifs", type=int, default=8)
    parser.add_argument(
        "--position_mode", choices=["fixed", "anywhere"], default="anywhere"
    )
    parser.add_argument(
        "--optimizer", choices=["quantile", "hybrid"], default="quantile"
    )
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--model_type", choices=["conv", "mlp"], default="conv")
    parser.add_argument(
        "--generator_output_norm",
        choices=["none", "l2", "centered_l2", "layernorm"],
        default="centered_l2",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ranker_list_size", type=int, default=128)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=256)
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
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--history_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/discrete_sequence_ranker_objectives"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    opt, objective, problem = build_optimizer(args)
    history = [record_buffer_history(opt, 0, objective.evaluation_count)]
    start = time.perf_counter()
    for iteration in range(1, args.n_iter + 1):
        opt.step()
        if args.history_interval > 0 and iteration % args.history_interval == 0:
            history.append(
                record_buffer_history(opt, iteration, objective.evaluation_count)
            )
    elapsed = time.perf_counter() - start

    best_value = float(opt.buffer.B.get_value(0))
    best_score = -best_value
    budget = objective.evaluation_count
    baseline_objective = MotifObjective(problem)
    random_best, random_curve = random_search(
        baseline_objective,
        budget=budget,
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"motif_l{args.length}_a{args.alphabet_size}_{args.position_mode}_"
        f"{args.optimizer}_iter{args.n_iter}_seed{args.seed}.npz"
    )
    history_array = np.asarray(
        [
            [
                row["iteration"],
                row["eval_count"],
                row["best_score"],
                row["mean_score"],
                row["median_score"],
                row["p10_score"],
                row["p90_score"],
            ]
            for row in history
        ],
        dtype=np.float32,
    )
    history_plot_path = output_path.with_suffix(".png")
    plot_history(history, history_plot_path, title=f"motif {args.optimizer}")
    np.savez_compressed(
        output_path,
        optimizer=np.asarray([args.optimizer]),
        length=np.asarray([args.length], dtype=np.int32),
        alphabet_size=np.asarray([args.alphabet_size], dtype=np.int32),
        position_mode=np.asarray([args.position_mode]),
        best_score=np.asarray([best_score], dtype=np.float32),
        best_value=np.asarray([best_value], dtype=np.float32),
        random_best=np.asarray([random_best], dtype=np.float32),
        random_curve=np.asarray(random_curve, dtype=np.float32),
        budget=np.asarray([budget], dtype=np.int32),
        elapsed=np.asarray([elapsed], dtype=np.float32),
        n_iter=np.asarray([args.n_iter], dtype=np.int32),
        batch_size=np.asarray([args.batch_size], dtype=np.int32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        generator_output_norm=np.asarray([args.generator_output_norm]),
        ranker_tau=np.asarray([args.ranker_tau], dtype=np.float32),
        utility_weight=np.asarray([args.utility_weight], dtype=np.float32),
        generator_utility_weight=np.asarray(
            [args.generator_utility_weight], dtype=np.float32
        ),
        history=history_array,
        history_columns=np.asarray(
            [
                "iteration",
                "eval_count",
                "best_score",
                "mean_score",
                "median_score",
                "p10_score",
                "p90_score",
            ]
        ),
        history_plot=np.asarray([str(history_plot_path)]),
    )
    print(f"optimizer={args.optimizer}")
    print(f"length={args.length}")
    print(f"alphabet_size={args.alphabet_size}")
    print(f"position_mode={args.position_mode}")
    print(f"generator_output_norm={args.generator_output_norm}")
    print(f"budget={budget}")
    print(f"best_score={best_score:.6f}")
    print(f"random_best={random_best:.6f}")
    print(f"elapsed={elapsed:.2f}")
    print(f"history_plot={history_plot_path}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
