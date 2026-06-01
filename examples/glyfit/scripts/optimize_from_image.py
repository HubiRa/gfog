#!/usr/bin/env python3
"""Optimize glyph fitting from a target image using gfog."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# Add gfog and glyfit to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gfog.buffer import Buffer
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import DefaultOpt, HingeGANOpt, WGANGPOpt, WGANOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda
from torch.nn import BCEWithLogitsLoss

from glyfit import GlyphEnv, build_topology, load_glyph_commands


def find_system_font() -> str:
    """Find a suitable TTF font on the system."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError("No suitable font found")


def load_target_image(path: str, width: int, height: int) -> np.ndarray:
    """Load and preprocess target image.

    Args:
        path: Path to image file.
        width: Target width.
        height: Target height.

    Returns:
        Grayscale image as numpy array (H, W), float32 in [0, 1].
    """
    img = Image.open(path)

    # Convert to grayscale
    if img.mode != "L":
        img = img.convert("L")

    # Resize to target dimensions
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Convert to numpy and normalize
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Invert if needed (we want white glyph on black background)
    # Check if image is mostly white (background) or mostly black
    if img_np.mean() > 0.5:
        img_np = 1.0 - img_np

    return img_np


class GlyphFitFunction:
    """Wrapper for GlyphEnv that conforms to gfog's function interface.

    The glyph environment already works in a delta-parameterization: the vector
    optimized by GFog is added to the base glyph control points. We therefore
    keep the generator output local by scaling and optionally bounding it.
    """

    def __init__(self, env: GlyphEnv, scale: float = 1.0):
        self.env = env
        self.input_dim = env.dim
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> list[float]:
        v_batch = x * self.scale
        losses = self.env.evaluate_batch(v_batch)
        return losses.tolist()


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


def main():
    parser = argparse.ArgumentParser(description="Optimize glyph fitting from target image")
    parser.add_argument(
        "experiment_dir", type=str, help="Path to experiment directory containing target.png"
    )
    parser.add_argument("--font", type=str, default=None, help="Path to TTF font file")
    parser.add_argument("--char", type=str, default="A", help="Character to use as base glyph")
    parser.add_argument("--width", type=int, default=64, help="Image width")
    parser.add_argument("--height", type=int, default=64, help="Image height")
    parser.add_argument("--n_iter", type=int, default=500, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument(
        "--param_scale",
        type=float,
        default=15.0,
        help="Max local glyph delta in font units. Smaller values keep search near the base glyph.",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=1,
        help="Subdivisions per curve segment (e.g. 200 for ~200x more points)",
    )
    parser.add_argument(
        "--curiosity",
        type=float,
        default=10.0,
        help="Curiosity loss weight (0 to disable)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "sdt", "combined"],
        help="Loss function type",
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=1.0,
        help="MSE weight for combined loss",
    )
    parser.add_argument(
        "--sdt_weight",
        type=float,
        default=1.0,
        help="SDT weight for combined loss",
    )
    parser.add_argument(
        "--smoothness",
        type=float,
        default=1e-4,
        help="Smoothness regularization weight (0 = disabled)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="wgangp",
        choices=["default", "hinge", "wgan", "wgangp"],
        help="GFog optimizer variant. WGANGP is the most stable default for this hard example.",
    )
    parser.add_argument(
        "--g_lr",
        type=float,
        default=0.001,
        help="Generator learning rate",
    )
    parser.add_argument(
        "--d_lr",
        type=float,
        default=0.005,
        help="Discriminator/critic learning rate",
    )
    parser.add_argument(
        "--discriminator_steps",
        type=int,
        default=3,
        help="Number of discriminator/critic updates per generator update",
    )
    parser.add_argument(
        "--gradient_penalty_weight",
        type=float,
        default=10.0,
        help="Gradient penalty weight for WGANGPOpt",
    )
    parser.add_argument(
        "--weight_clip",
        type=float,
        default=0.01,
        help="Weight clipping value for WGANOpt",
    )
    args = parser.parse_args()

    # Setup paths
    experiment_dir = Path(args.experiment_dir)
    target_path = experiment_dir / "target.png"

    if not target_path.exists():
        print(f"Error: target.png not found in {experiment_dir}")
        sys.exit(1)

    # Font path
    ttf_path = args.font if args.font else find_system_font()

    print(f"Experiment dir: {experiment_dir}")
    print(f"Target image: {target_path}")
    print(f"Font: {ttf_path}")
    print(f"Character: '{args.char}'")
    print(f"Resolution: {args.width}x{args.height}")

    # Load target image
    print("\nLoading target image...")
    target_np = load_target_image(str(target_path), args.width, args.height)
    print(f"Target shape: {target_np.shape}")
    print(f"Target range: [{target_np.min():.3f}, {target_np.max():.3f}]")

    # Load base glyph topology
    print("\nLoading base glyph topology...")
    commands = load_glyph_commands(ttf_path, args.char)
    topology = build_topology(commands, subdivisions=args.subdivisions)
    print(f"Base glyph has {topology.base_points.shape[0]} control points")
    if args.subdivisions > 1:
        print(f"(subdivided {args.subdivisions}x per segment)")

    # Create environment with the target image
    env = GlyphEnv(
        ttf_path=ttf_path,
        char=args.char,
        target_image_np=target_np,
        width=args.width,
        height=args.height,
        subdivisions=args.subdivisions,
        loss_type=args.loss,
        mse_weight=args.mse_weight,
        sdt_weight=args.sdt_weight,
        smoothness_weight=args.smoothness,
    )

    F_DIM = env.dim
    print(f"Parameter dimension: {F_DIM}")
    print(f"Loss type: {args.loss}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Parameter delta scale: ±{args.param_scale}")
    print("Generator output is bounded with tanh, so optimization stays local to the base glyph.")
    if args.smoothness > 0:
        print(f"Smoothness weight: {args.smoothness}")

    # Create function wrapper
    test_function = GlyphFitFunction(env, scale=args.param_scale)

    # Setup gfog components
    DEVICE = torch.device("cpu")

    fn = components.Fn(
        f=test_function,
        input_dim=F_DIM,
        device=DEVICE,
        dtype=torch.float,
    )

    # Networks - scale with parameter dimension
    # For high-dimensional problems, use larger networks
    # if F_DIM > 1000:
    #     g_hidden = [512, 512, 256]
    #     d_hidden = [256, 256, 128]
    # elif F_DIM > 100:
    #     g_hidden = [256, 256, 128]
    #     d_hidden = [128, 128, 64]
    # else:
    #     g_hidden = [128, 128, 64]
    #     d_hidden = [128, 128, 64]

    g_hidden = [128, 64]
    d_hidden = [128, 64]

    G = MLP(
        input_dim=args.latent_dim,
        output_dim=F_DIM,
        hidden_dims=g_hidden,
        output_activation=torch.nn.Tanh(),
    ).to(DEVICE)

    D = MLP(
        input_dim=F_DIM,
        output_dim=1,
        hidden_dims=d_hidden,
        use_spectral_norm=args.optimizer in {"default", "hinge"},
    ).to(DEVICE)

    print(f"Generator hidden layers: {g_hidden}")
    print(f"Discriminator hidden layers: {d_hidden}")

    buffer = components.BufferComp(B=Buffer(buffer_size=2 * args.batch_size))

    # Setup curiosity loss (can be disabled with --curiosity 0)
    if args.curiosity > 0:
        curiosity_loss = WangIsolaUniformity(
            config=WangIsolaUniformityConfig(use_buffer=True, weight=args.curiosity),
            buffer=buffer.B,
        )
    else:
        curiosity_loss = None

    gan = components.GAN(
        G=G,
        D=D,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
        latent_dim=args.latent_dim,
        optimizerG=torch.optim.Adam(lr=args.g_lr, params=G.parameters()),
        optimizerD=torch.optim.Adam(lr=args.d_lr, params=D.parameters()),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d), b=args.batch_size, d=args.latent_dim
        ),
        device=DEVICE,
        dtype=torch.float,
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

    # Progress tracking
    progress = Progress(
        TextColumn("Iter {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("Best: {task.fields[best]:.6f}"),
        TextColumn("Mean: {task.fields[mean]:.6f}"),
        TimeElapsedColumn(),
    )

    # Optimization loop
    print(f"\nStarting optimization for {args.n_iter} iterations...")
    best_values = []

    with progress:
        task = progress.add_task("Optimizing", total=args.n_iter, best=999.0, mean=999.0)

        for i in range(args.n_iter):
            optimizer.step()

            best_val = buffer.B.get_value(0)
            mean_val = buffer.B.get_mean_buffer_value()
            best_values.append(best_val)

            progress.update(task, advance=1, best=best_val, mean=mean_val)

    # Results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)

    best_params = buffer.B.get_top_k(1).squeeze()
    best_loss = buffer.B.get_value(0)

    print(f"Best loss: {best_loss:.6f}")
    print(f"Final mean buffer loss: {buffer.B.get_mean_buffer_value():.6f}")

    # Save results
    # Save preprocessed target (what the optimizer actually sees)
    target_processed = Image.fromarray((target_np * 255).astype(np.uint8))
    target_processed.save(experiment_dir / "target_processed.png")
    print(f"\nSaved processed target to {experiment_dir}/target_processed.png")

    # Save best result
    best_V = best_params * args.param_scale
    best_img_np = env.render(best_V)
    best_img = Image.fromarray((best_img_np * 255).astype(np.uint8))
    best_img.save(experiment_dir / "best.png")
    print(f"Saved best result to {experiment_dir}/best.png")

    # Save comparison
    comparison = np.hstack([target_np, best_img_np])
    comparison_img = Image.fromarray((comparison * 255).astype(np.uint8))
    comparison_img.save(experiment_dir / "comparison.png")
    print(f"Saved comparison to {experiment_dir}/comparison.png")

    # Save loss curve
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(best_values, linewidth=0.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Loss (MSE)")
        ax.set_title(f"Glyph Fitting - {experiment_dir.name}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.savefig(experiment_dir / "loss_curve.png", dpi=150)
        plt.close(fig)
        print(f"Saved loss curve to {experiment_dir}/loss_curve.png")
    except ImportError:
        pass

    # Save config
    config = {
        "font": ttf_path,
        "char": args.char,
        "width": args.width,
        "height": args.height,
        "n_iter": args.n_iter,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "param_scale": args.param_scale,
        "subdivisions": args.subdivisions,
        "curiosity": args.curiosity,
        "optimizer": args.optimizer,
        "g_lr": args.g_lr,
        "d_lr": args.d_lr,
        "discriminator_steps": args.discriminator_steps,
        "gradient_penalty_weight": args.gradient_penalty_weight,
        "weight_clip": args.weight_clip,
        "loss_type": args.loss,
        "mse_weight": args.mse_weight,
        "sdt_weight": args.sdt_weight,
        "smoothness": args.smoothness,
        "best_loss": best_loss,
        "final_mean_loss": buffer.B.get_mean_buffer_value(),
        "param_dim": F_DIM,
        "num_points": env.num_points,
        "timestamp": datetime.now().isoformat(),
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {experiment_dir}/config.json")

    # Save best parameters
    np.save(experiment_dir / "best_params.npy", best_V.detach().cpu().numpy())
    print(f"Saved best parameters to {experiment_dir}/best_params.npy")


if __name__ == "__main__":
    main()
