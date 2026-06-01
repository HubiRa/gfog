import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import DefaultOpt, HingeGANOpt, WGANGPOpt, WGANOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


def get_optimizer_cls(name: str):
    optimizers = {
        "default": DefaultOpt,
        "hinge": HingeGANOpt,
        "wgan": WGANOpt,
        "wgangp": WGANGPOpt,
    }
    try:
        return optimizers[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer '{name}'. Expected one of {sorted(optimizers)}"
        ) from exc


class LinearContinuousPolicyEvaluator:
    """Evaluate a linear tanh policy on a continuous-control environment."""

    def __init__(
        self,
        env_id: str,
        obs_dim: int,
        action_dim: int,
        episode_steps: int,
        runs_per_env: int = 1,
        seed: int = 0,
    ) -> None:
        self.env_id = env_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_steps = episode_steps
        self.runs_per_env = runs_per_env
        self.seed = seed

    def _rollout(self, theta: np.ndarray, rollout_seed: int) -> float:
        env = gym.make(self.env_id, max_episode_steps=self.episode_steps)
        try:
            weight_size = self.obs_dim * self.action_dim
            w = theta[:weight_size].reshape(self.obs_dim, self.action_dim)
            b = theta[weight_size : weight_size + self.action_dim]

            obs, _ = env.reset(seed=rollout_seed)
            total_reward = 0.0
            done = False
            while not done:
                action = np.tanh(obs @ w + b)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            return float(total_reward)
        finally:
            env.close()

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[float]:
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()

        returns = []
        for i, candidate in enumerate(theta):
            runs = [
                self._rollout(candidate, self.seed + 10_000 * i + j)
                for j in range(self.runs_per_env)
            ]
            returns.append(-float(np.median(runs)))
        return returns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GFog HalfCheetah example")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="wgangp",
        choices=["default", "hinge", "wgan", "wgangp"],
        help="Optimizer variant. WGANGP is the recommended default for this harder RL example.",
    )
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v5")
    parser.add_argument(
        "--n_iter", type=int, default=100, help="Optimization iterations"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--episode_steps", type=int, default=500)
    parser.add_argument("--runs_per_env", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curiosity", type=float, default=2.0)
    parser.add_argument("--g_lr", type=float, default=0.005)
    parser.add_argument("--d_lr", type=float, default=0.01)
    parser.add_argument("--discriminator_steps", type=int, default=3)
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument(
        "--generalization_steps",
        type=int,
        default=1000,
        help="Episode length used for the final generalization check",
    )
    return parser


def run_experiment(
    args: argparse.Namespace, *, show_tables: bool = True
) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe_env = gym.make(args.env_id, max_episode_steps=args.episode_steps)
    try:
        obs_dim = int(np.prod(probe_env.observation_space.shape))
        action_dim = int(np.prod(probe_env.action_space.shape))
    finally:
        probe_env.close()

    f_dim = obs_dim * action_dim + action_dim
    device = torch.device("cpu")

    fn = components.Fn(
        f=LinearContinuousPolicyEvaluator(
            args.env_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            episode_steps=args.episode_steps,
            runs_per_env=args.runs_per_env,
            seed=args.seed,
        ),
        input_dim=f_dim,
        device=device,
        dtype=torch.float32,
    )

    g = MLP(
        input_dim=args.latent_dim,
        output_dim=f_dim,
        hidden_dims=[128, 64],
        output_activation=torch.nn.Tanh(),
    ).to(device)
    d = MLP(
        input_dim=f_dim,
        output_dim=1,
        hidden_dims=[128, 64],
        use_spectral_norm=args.optimizer in {"default", "hinge"},
    ).to(device)

    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=2 * args.batch_size, value_levels=Levels(["median return"])
        )
    )

    curiosity_loss = None
    if args.curiosity > 0:
        curiosity_loss = WangIsolaUniformity(
            config=WangIsolaUniformityConfig(use_buffer=True, weight=args.curiosity),
            buffer=buffer.B,
        )

    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
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

    optimizer_cls = get_optimizer_cls(args.optimizer)
    optimizer = optimizer_cls(
        components.OptComponents(
            fn=fn,
            gan=gan,
            batch_size=args.batch_size,
            buffer=buffer,
            discriminator_steps=args.discriminator_steps,
            elite_sampling="random_top_k",
            elite_pool_size=2 * args.batch_size,
            weight_clip=args.weight_clip if args.optimizer == "wgan" else None,
            gradient_penalty_weight=args.gradient_penalty_weight,
        )
    )

    initial_best = buffer.B.get_value(0)
    initial_mean = buffer.B.get_mean_buffer_value()

    logger.info(
        f"HalfCheetah config: optimizer={args.optimizer} env={args.env_id} "
        f"n_iter={args.n_iter} batch_size={args.batch_size} runs_per_env={args.runs_per_env} seed={args.seed}"
    )
    logger.info(
        f"Policy parameterization: obs_dim={obs_dim} action_dim={action_dim} f_dim={f_dim}"
    )
    logger.info(
        f"Initial best={initial_best:.4f} mean={initial_mean:.4f} "
        f"(remember: more negative = better because we minimize negative return)"
    )
    if show_tables:
        logger.info("Initial top-5 buffer values")
        buffer.B.print_values(slice(0, 5, 1))

    optimizer.optimize(n_iter=args.n_iter, verbose=True)

    final_best = buffer.B.get_value(0)
    final_mean = buffer.B.get_mean_buffer_value()
    improvement = final_best - initial_best

    logger.info(
        f"Final best={final_best:.4f} mean={final_mean:.4f} improvement={improvement:.4f}"
    )
    if show_tables:
        logger.info("Top-5 buffer values after optimization")
        buffer.B.print_values(slice(0, 5, 1))

    test_evaluator = LinearContinuousPolicyEvaluator(
        args.env_id,
        obs_dim=obs_dim,
        action_dim=action_dim,
        episode_steps=args.generalization_steps,
        runs_per_env=max(3, args.runs_per_env),
        seed=args.seed + 123_456,
    )
    test_results = test_evaluator(buffer.B.get_top_k(5))
    generalization_best = min(test_results)
    logger.info(
        f"Generalization check at {args.generalization_steps} steps (top 5): {test_results}"
    )
    logger.info(f"Best generalization score: {generalization_best:.4f}")

    return {
        "optimizer": args.optimizer,
        "seed": args.seed,
        "curiosity": args.curiosity,
        "initial_best": initial_best,
        "initial_mean": initial_mean,
        "final_best": final_best,
        "final_mean": final_mean,
        "improvement": improvement,
        "generalization_best": generalization_best,
        "generalization_top5": test_results,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "f_dim": f_dim,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args, show_tables=True)


if __name__ == "__main__":
    main()
