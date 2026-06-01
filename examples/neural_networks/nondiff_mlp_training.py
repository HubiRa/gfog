"""Train a tiny neural net with GFog through a non-differentiable objective.

The optimized vector is the flattened parameter vector of a small MLP. The black
box quantizes those parameters, runs a hard-threshold classifier, and returns
0/1 classification error. No gradient through the task loss is available to G.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import DefaultOpt, LSGANOpt, WGANOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


@dataclass(frozen=True)
class TinyMLPShape:
    input_dim: int
    hidden_dim: int
    output_dim: int = 1

    @property
    def n_params(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )


class NonDifferentiableMLPObjective:
    """Black-box objective over flattened MLP parameters."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        shape: TinyMLPShape,
        weight_scale: float = 2.0,
        quantization_levels: int = 0,
        l2_weight: float = 1e-4,
    ) -> None:
        self.x = x.to(torch.float32)
        self.y = y.to(torch.float32)
        self.shape = shape
        self.weight_scale = weight_scale
        self.quantization_levels = quantization_levels
        self.l2_weight = l2_weight

    def _decode(self, params: torch.Tensor) -> torch.Tensor:
        weights = self.weight_scale * torch.tanh(params)
        if self.quantization_levels > 1:
            levels = self.quantization_levels - 1
            weights = torch.round(
                (weights + self.weight_scale) / (2 * self.weight_scale) * levels
            )
            weights = weights / levels * (2 * self.weight_scale) - self.weight_scale
        return weights

    def _forward(self, weights: torch.Tensor) -> torch.Tensor:
        offset = 0
        w1_size = self.shape.input_dim * self.shape.hidden_dim
        w1 = weights[offset : offset + w1_size].reshape(
            self.shape.input_dim,
            self.shape.hidden_dim,
        )
        offset += w1_size
        b1 = weights[offset : offset + self.shape.hidden_dim]
        offset += self.shape.hidden_dim
        w2_size = self.shape.hidden_dim * self.shape.output_dim
        w2 = weights[offset : offset + w2_size].reshape(
            self.shape.hidden_dim,
            self.shape.output_dim,
        )
        offset += w2_size
        b2 = weights[offset : offset + self.shape.output_dim]
        hidden = torch.tanh(self.x @ w1 + b1)
        return (hidden @ w2 + b2).reshape(-1)

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        values = []
        for params in candidates.detach().cpu():
            weights = self._decode(params)
            logits = self._forward(weights)
            pred = (logits >= 0).to(torch.float32)
            error = (pred != self.y).to(torch.float32).mean()
            l2 = self.l2_weight * torch.mean(weights * weights)
            values.append(error + l2)
        return torch.stack(values).to(candidates.device, candidates.dtype)

    def accuracy(self, params: torch.Tensor) -> float:
        weights = self._decode(params.detach().cpu())
        logits = self._forward(weights)
        pred = (logits >= 0).to(torch.float32)
        return float((pred == self.y).to(torch.float32).mean().item())


def make_moons_dataset(
    n: int, noise: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    half = n // 2
    theta0 = torch.rand(half, generator=gen) * torch.pi
    theta1 = torch.rand(n - half, generator=gen) * torch.pi
    x0 = torch.stack([torch.cos(theta0), torch.sin(theta0)], dim=1)
    x1 = torch.stack([1.0 - torch.cos(theta1), 0.5 - torch.sin(theta1)], dim=1)
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(half), torch.ones(n - half)], dim=0)
    x = x + noise * torch.randn(x.shape, generator=gen)
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)
    order = torch.randperm(n, generator=gen)
    return x[order], y[order]


def build_optimizer(
    args: argparse.Namespace,
) -> tuple[DefaultOpt | LSGANOpt | WGANOpt, NonDifferentiableMLPObjective]:
    device = torch.device("cpu")
    shape = TinyMLPShape(input_dim=2, hidden_dim=args.task_hidden_dim)
    x, y = make_moons_dataset(args.n_samples, args.noise, args.seed)
    objective = NonDifferentiableMLPObjective(
        x=x,
        y=y,
        shape=shape,
        weight_scale=args.task_weight_scale,
        quantization_levels=args.quantization_levels,
        l2_weight=args.l2_weight,
    )
    fn = components.Fn(
        f=objective,
        input_dim=shape.n_params,
        device=device,
        dtype=torch.float32,
    )
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
    )
    g = MLP(
        input_dim=args.latent_dim,
        output_dim=shape.n_params,
        hidden_dims=[args.generator_hidden_dim, args.generator_hidden_dim],
    ).to(device)
    d = MLP(
        input_dim=shape.n_params,
        output_dim=1,
        hidden_dims=[args.discriminator_hidden_dim, args.discriminator_hidden_dim],
    ).to(device)
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
        weight_clip=args.weight_clip if args.optimizer == "wgan" else None,
    )
    opt_cls: type[DefaultOpt | LSGANOpt | WGANOpt]
    if args.optimizer == "default":
        opt_cls = DefaultOpt
    elif args.optimizer == "lsgan":
        opt_cls = LSGANOpt
    elif args.optimizer == "wgan":
        opt_cls = WGANOpt
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return opt_cls(opt_components), objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--generator_hidden_dim", type=int, default=128)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=128)
    parser.add_argument("--task_hidden_dim", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--task_weight_scale", type=float, default=2.0)
    parser.add_argument(
        "--quantization_levels",
        type=int,
        default=8,
        help="Quantize task-network weights to this many levels; 0 disables quantization.",
    )
    parser.add_argument("--l2_weight", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", choices=["default", "lsgan", "wgan"], default="default"
    )
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=3e-3)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/nondiff_mlp_training"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer, objective = build_optimizer(args)
    optimizer.optimize(args.n_iter, verbose=True)

    best = optimizer.buffer.B.get_top_k(1).squeeze(0)
    best_value = float(optimizer.buffer.B.get_value(0))
    best_accuracy = objective.accuracy(best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"nondiff_mlp_seed_{args.seed}.npz",
        best_params=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_accuracy=np.asarray([best_accuracy], dtype=np.float32),
        sorted_values=np.asarray(
            optimizer.buffer.B.get_sorted_values(), dtype=np.float32
        ),
        quantization_levels=np.asarray([args.quantization_levels], dtype=np.int32),
    )
    print(f"best_value={best_value:.6f}")
    print(f"best_accuracy={best_accuracy:.4f}")
    print(f"saved={args.output_dir / f'nondiff_mlp_seed_{args.seed}.npz'}")


if __name__ == "__main__":
    main()
