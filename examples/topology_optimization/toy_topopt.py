import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import DefaultOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


@dataclass
class ToyTopOptConfig:
    grid_size: int = 16
    volume_max: float = 0.32
    roughness_max: float = 0.34
    reward_outside_template: float = 0.35


class ToyTopologyEvaluator:
    """Cheap multimodal topology surrogate with explicit constraints."""

    def __init__(self, config: ToyTopOptConfig) -> None:
        self.config = config
        self.templates = self._build_templates(config.grid_size)

    def _build_templates(self, n: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(n, dtype=torch.float32),
            torch.arange(n, dtype=torch.float32),
            indexing="ij",
        )
        xx = xx / max(n - 1, 1)
        yy = yy / max(n - 1, 1)

        center = torch.exp(-((yy - 0.5) ** 2) / 0.012)

        upper_curve = 0.28 + 0.18 * torch.cos(np.pi * xx)
        upper = torch.exp(-((yy - upper_curve) ** 2) / 0.010)

        lower_curve = 0.72 - 0.18 * torch.cos(np.pi * xx)
        lower = torch.exp(-((yy - lower_curve) ** 2) / 0.010)

        diag_up = torch.exp(-((yy - (0.70 - 0.45 * xx)) ** 2) / 0.012)
        diag_down = torch.exp(-((yy - (0.30 + 0.45 * xx)) ** 2) / 0.012)

        left_anchor = torch.exp(-((xx - 0.02) ** 2) / 0.002) * torch.exp(
            -((yy - 0.50) ** 2) / 0.030
        )
        right_anchor = torch.exp(-((xx - 0.98) ** 2) / 0.002) * torch.exp(
            -((yy - 0.50) ** 2) / 0.030
        )
        anchors = torch.clamp(left_anchor + right_anchor, 0.0, 1.0)

        templates = []
        for raw in [center, upper, lower, diag_up, diag_down]:
            template = torch.clamp(0.75 * raw + 0.9 * anchors, 0.0, 1.0)
            template = template / template.max().clamp_min(1e-8)
            templates.append(template)
        return torch.stack(templates)

    def _roughness(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean(dim=(1, 2))
        dy = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean(dim=(1, 2))
        return 0.5 * (dx + dy)

    def _best_template_score(self, x: torch.Tensor) -> torch.Tensor:
        templates = self.templates.to(device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(1)
        inside = (x_expanded * templates.unsqueeze(0)).mean(dim=(2, 3))
        outside = (x_expanded * (1.0 - templates.unsqueeze(0))).mean(dim=(2, 3))
        scores = inside - self.config.reward_outside_template * outside
        return scores.max(dim=1).values

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[list[float]]:
        if isinstance(theta, np.ndarray):
            x = torch.from_numpy(theta).to(torch.float32)
        else:
            x = theta.detach().to(torch.float32).cpu()

        x = x.reshape(-1, self.config.grid_size, self.config.grid_size)
        volume = x.mean(dim=(1, 2))
        roughness = self._roughness(x)
        structure_score = self._best_template_score(x)

        volume_violation = torch.clamp(volume - self.config.volume_max, min=0.0)
        roughness_violation = torch.clamp(
            roughness - self.config.roughness_max,
            min=0.0,
        )
        objective = -structure_score

        values = torch.stack(
            [volume_violation, roughness_violation, objective],
            dim=1,
        )
        return values.tolist()


def pairwise_l2_mean(designs: torch.Tensor) -> float:
    flat = designs.reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = flat[:, None, :] - flat[None, :, :]
    dists = diffs.pow(2).sum(dim=-1).sqrt()
    triu = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    return float(dists[triu[0], triu[1]].mean().item())


def pairwise_hamming_mean(designs: torch.Tensor, threshold: float = 0.5) -> float:
    flat = (designs >= threshold).reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = (flat[:, None, :] != flat[None, :, :]).float().mean(dim=-1)
    triu = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    return float(diffs[triu[0], triu[1]].mean().item())


def save_design_grid(
    designs: torch.Tensor,
    values: list[list[float]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_designs = designs.shape[0]
    cols = min(3, num_designs)
    rows = int(np.ceil(num_designs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= num_designs:
            ax.axis("off")
            continue
        ax.imshow(designs[idx].cpu().numpy(), cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        vol_violation, roughness_violation, objective = values[idx]
        ax.set_title(
            f"#{idx + 1}\nvol={vol_violation:.3f} rough={roughness_violation:.3f} obj={objective:.3f}",
            fontsize=10,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GFog toy topology optimization")
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--n_iter", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--curiosity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g_lr", type=float, default=0.01)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument("--discriminator_steps", type=int, default=3)
    parser.add_argument("--volume_max", type=float, default=0.32)
    parser.add_argument("--roughness_max", type=float, default=0.34)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/toy_topopt"),
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ToyTopOptConfig(
        grid_size=args.grid_size,
        volume_max=args.volume_max,
        roughness_max=args.roughness_max,
    )
    evaluator = ToyTopologyEvaluator(config)

    f_dim = args.grid_size * args.grid_size
    device = torch.device("cpu")

    fn = components.Fn(
        f=evaluator,
        input_dim=f_dim,
        device=device,
        dtype=torch.float32,
    )

    g = MLP(
        input_dim=args.latent_dim,
        output_dim=f_dim,
        hidden_dims=[128, 128],
        output_activation=torch.nn.Sigmoid(),
    ).to(device)
    d = MLP(
        input_dim=f_dim,
        output_dim=1,
        hidden_dims=[128, 128],
        use_spectral_norm=True,
    ).to(device)

    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=Levels(
                ["volume_violation", "roughness_violation", "objective"]
            ),
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

    optimizer = DefaultOpt(
        components.OptComponents(
            fn=fn,
            gan=gan,
            batch_size=args.batch_size,
            buffer=buffer,
            discriminator_steps=args.discriminator_steps,
            elite_sampling="random_top_k",
            elite_pool_size=args.buffer_multiplier * args.batch_size,
        )
    )

    logger.info(
        f"ToyTopOpt config: grid={args.grid_size}x{args.grid_size} n_iter={args.n_iter} "
        f"batch_size={args.batch_size} latent_dim={args.latent_dim} curiosity={args.curiosity} seed={args.seed}"
    )
    logger.info(
        f"Constraint limits: volume_max={args.volume_max:.3f} roughness_max={args.roughness_max:.3f}"
    )
    logger.info("Initial top-5 buffer values")
    buffer.B.print_values(slice(0, 5, 1))

    optimizer.optimize(args.n_iter, verbose=True)

    logger.info("Final top-5 buffer values")
    buffer.B.print_values(slice(0, 5, 1))

    top_k = min(9, len(buffer.B))
    top_designs = buffer.B.get_top_k(top_k).reshape(
        top_k, args.grid_size, args.grid_size
    )
    top_values = buffer.B.get_sorted_values()[:top_k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"curiosity_{args.curiosity:g}_seed_{args.seed}"
    save_design_grid(
        top_designs,
        top_values,
        args.output_dir / f"top_designs_{suffix}.png",
        title=f"Toy TopOpt Top Designs (curiosity={args.curiosity:g}, seed={args.seed})",
    )
    save_design_grid(
        evaluator.templates,
        [[0.0, 0.0, float(-i)] for i in range(evaluator.templates.shape[0])],
        args.output_dir / "templates.png",
        title="Toy TopOpt Template Families",
    )
    np.savez_compressed(
        args.output_dir / f"top_designs_{suffix}.npz",
        designs=top_designs.cpu().numpy(),
        values=np.asarray(top_values, dtype=np.float32),
        templates=evaluator.templates.cpu().numpy(),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        seed=np.asarray([args.seed], dtype=np.int32),
    )

    return {
        "best_value": top_values[0],
        "best_objective": float(top_values[0][2]),
        "mean_objective_topk": float(np.mean([row[2] for row in top_values])),
        "mean_l2": pairwise_l2_mean(top_designs),
        "mean_hamming": pairwise_hamming_mean(top_designs),
        "curiosity": args.curiosity,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "top_k": top_k,
        "artifact_path": str(args.output_dir / f"top_designs_{suffix}.npz"),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(args)
    logger.info(f"Saved outputs to {result['output_dir']}")
    logger.info(f"Saved raw artifacts to {result['artifact_path']}")
    logger.info(f"Best buffer value vector: {result['best_value']}")
    logger.info(
        f"Top-k summary: best_objective={result['best_objective']:.4f} "
        f"mean_l2={result['mean_l2']:.4f} mean_hamming={result['mean_hamming']:.4f}"
    )


if __name__ == "__main__":
    main()
