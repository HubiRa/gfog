"""Compare quantile and hybrid ranker objectives on Gymnasium RL tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cartpole import CartPoleEvaluator  # noqa: E402
from halfcheetah import LinearContinuousPolicyEvaluator  # noqa: E402
import gymnasium as gym  # noqa: E402
from gfog.buffer import Buffer, Levels  # noqa: E402
from gfog.models import MLP  # noqa: E402
from gfog.opt import (  # noqa: E402
    HybridContextualUtilityRankerOpt,
    QuantileRankedDefaultOpt,
    components,
    make_torch_optimizer,
)
from gfog.opt.latents_sampler import LatentSamplerLambda  # noqa: E402


def build_optimizer(args: argparse.Namespace):
    if args.task == "cartpole":
        f_dim = 5
        evaluator = CartPoleEvaluator(
            episode_steps=args.episode_steps,
            runs_per_env=args.runs_per_env,
            seed=args.seed,
        )
        g_hidden_dims = [args.hidden_dim]
        d_hidden_dims = [args.hidden_dim]
    elif args.task == "halfcheetah":
        probe_env = gym.make(args.env_id, max_episode_steps=args.episode_steps)
        try:
            obs_dim = int(np.prod(probe_env.observation_space.shape))
            action_dim = int(np.prod(probe_env.action_space.shape))
        finally:
            probe_env.close()
        f_dim = obs_dim * action_dim + action_dim
        evaluator = LinearContinuousPolicyEvaluator(
            args.env_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            episode_steps=args.episode_steps,
            runs_per_env=args.runs_per_env,
            seed=args.seed,
        )
        g_hidden_dims = [args.hidden_dim, max(args.hidden_dim // 2, 16)]
        d_hidden_dims = [args.hidden_dim, max(args.hidden_dim // 2, 16)]
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    device = torch.device("cpu")
    g = MLP(
        input_dim=args.latent_dim,
        output_dim=f_dim,
        hidden_dims=g_hidden_dims,
        output_activation=torch.nn.Tanh(),
    ).to(device)
    d = MLP(
        input_dim=f_dim,
        output_dim=1,
        hidden_dims=d_hidden_dims,
        use_spectral_norm=args.discriminator_spectral_norm,
    ).to(device)
    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=Levels(["negative median return"]),
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
            lambda b, d: torch.randn(b, d) * args.latent_scale,
            b=args.batch_size,
            d=args.latent_dim,
        ),
        device=device,
        dtype=torch.float32,
    )
    opt_components = components.OptComponents(
        fn=components.Fn(
            f=evaluator,
            input_dim=f_dim,
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
    return opt


def record_buffer_history(opt, iteration: int) -> dict[str, float]:
    values = opt.buffer.B.get_sorted_values()
    objective = np.asarray([row[-1] for row in values], dtype=np.float64)
    returns = -objective
    return {
        "iteration": float(iteration),
        "eval_count": float(
            opt.buffer.B.buffer_size + iteration * opt.components.batch_size
        ),
        "best_return": float(returns[0]),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "p10_return": float(np.percentile(returns, 10)),
        "p90_return": float(np.percentile(returns, 90)),
    }


def plot_history(
    history: list[dict[str, float]], output_path: Path, title: str
) -> None:
    if not history:
        return
    evals = np.asarray([row["eval_count"] for row in history])
    best = np.asarray([row["best_return"] for row in history])
    mean = np.asarray([row["mean_return"] for row in history])
    median = np.asarray([row["median_return"] for row in history])
    p10 = np.asarray([row["p10_return"] for row in history])
    p90 = np.asarray([row["p90_return"] for row in history])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(evals, best, label="best buffer return", linewidth=2)
    ax.plot(evals, mean, label="mean buffer return", linewidth=1.5)
    ax.plot(evals, median, label="median buffer return", linewidth=1.5)
    ax.fill_between(evals, p10, p90, alpha=0.18, label="p10-p90 buffer range")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("median return, higher is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_generalization(
    opt,
    *,
    task: str,
    env_id: str,
    episode_steps: int,
    runs_per_env: int,
    seed: int,
) -> list[float]:
    if task == "cartpole":
        evaluator = CartPoleEvaluator(
            episode_steps=episode_steps,
            runs_per_env=runs_per_env,
            seed=seed,
        )
    elif task == "halfcheetah":
        probe_env = gym.make(env_id, max_episode_steps=episode_steps)
        try:
            obs_dim = int(np.prod(probe_env.observation_space.shape))
            action_dim = int(np.prod(probe_env.action_space.shape))
        finally:
            probe_env.close()
        evaluator = LinearContinuousPolicyEvaluator(
            env_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            episode_steps=episode_steps,
            runs_per_env=runs_per_env,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return [-float(v) for v in evaluator(opt.buffer.B.get_top_k(5))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["cartpole", "halfcheetah"], default="cartpole"
    )
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument(
        "--optimizer", choices=["quantile", "hybrid"], default="quantile"
    )
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--latent_scale", type=float, default=2.0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--runs_per_env", type=int, default=5)
    parser.add_argument("--generalization_steps", type=int, default=5000)
    parser.add_argument("--generalization_runs", type=int, default=10)
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.1)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--discriminator_spectral_norm", action="store_true")
    parser.add_argument("--ranker_list_size", type=int, default=16)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=32)
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
        default=Path("results/gymnasium_ranker_objectives"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    opt = build_optimizer(args)
    history = [record_buffer_history(opt, 0)]
    for iteration in range(1, args.n_iter + 1):
        opt.step()
        if args.history_interval > 0 and iteration % args.history_interval == 0:
            history.append(record_buffer_history(opt, iteration))

    best_negative_return = float(opt.buffer.B.get_value(0))
    best_return = -best_negative_return
    generalization_top5 = evaluate_generalization(
        opt,
        task=args.task,
        env_id=args.env_id,
        episode_steps=args.generalization_steps,
        runs_per_env=args.generalization_runs,
        seed=args.seed + 123_456,
    )
    generalization_best = max(generalization_top5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"{args.task}_{args.optimizer}_iter{args.n_iter}_seed{args.seed}.npz"
    )
    history_array = np.asarray(
        [
            [
                row["iteration"],
                row["eval_count"],
                row["best_return"],
                row["mean_return"],
                row["median_return"],
                row["p10_return"],
                row["p90_return"],
            ]
            for row in history
        ],
        dtype=np.float32,
    )
    history_plot_path = output_path.with_suffix(".png")
    plot_history(history, history_plot_path, title=f"{args.task}: {args.optimizer}")
    np.savez_compressed(
        output_path,
        task=np.asarray([args.task]),
        optimizer=np.asarray([args.optimizer]),
        best_return=np.asarray([best_return], dtype=np.float32),
        best_negative_return=np.asarray([best_negative_return], dtype=np.float32),
        generalization_best=np.asarray([generalization_best], dtype=np.float32),
        generalization_top5=np.asarray(generalization_top5, dtype=np.float32),
        n_iter=np.asarray([args.n_iter], dtype=np.int32),
        batch_size=np.asarray([args.batch_size], dtype=np.int32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        episode_steps=np.asarray([args.episode_steps], dtype=np.int32),
        runs_per_env=np.asarray([args.runs_per_env], dtype=np.int32),
        history=history_array,
        history_columns=np.asarray(
            [
                "iteration",
                "eval_count",
                "best_return",
                "mean_return",
                "median_return",
                "p10_return",
                "p90_return",
            ]
        ),
        history_plot=np.asarray([str(history_plot_path)]),
    )
    print(f"task={args.task}")
    print(f"optimizer={args.optimizer}")
    print(f"g_optimizer={args.g_optimizer}")
    print(f"d_optimizer={args.d_optimizer}")
    print(f"g_lr={args.g_lr}")
    print(f"d_lr={args.d_lr}")
    print(f"best_return={best_return:.4f}")
    print(f"generalization_best={generalization_best:.4f}")
    print(f"generalization_top5={generalization_top5}")
    print(f"history_plot={history_plot_path}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
