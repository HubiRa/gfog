#!/usr/bin/env python3
"""Optimize glyph fitting using gfog gradient-free optimizer."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

# Add gfog and glyfit to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gfog.buffer import Buffer
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import DefaultOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda
from torch.nn import BCEWithLogitsLoss

from glyfit import GlyphEnv, apply_vector, build_topology, load_glyph_commands, rasterize


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


class GlyphFitFunction:
    """Wrapper for GlyphEnv that conforms to gfog's function interface."""

    def __init__(self, env: GlyphEnv, scale: float = 1.0):
        """
        Args:
            env: GlyphEnv instance.
            scale: Scale factor for the parameter space (maps from normalized to font units).
        """
        self.env = env
        self.input_dim = env.dim
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> list[float]:
        """Evaluate batch of candidates.

        Args:
            x: Tensor of shape (B, D) with normalized parameters.

        Returns:
            List of loss values (one per candidate).
        """
        # Scale from normalized space to font units
        V_batch = x * self.scale

        # Evaluate using GlyphEnv
        losses = self.env.evaluate_batch(V_batch)

        # Return as list for gfog buffer insertion
        return losses.tolist()


def main():
    # Configuration
    CHAR = "A"
    WIDTH, HEIGHT = 64, 64  # Lower resolution for faster iteration
    N_ITER = 500
    BATCH_SIZE = 32
    LATENT_DIM = 32  # Latent dimension for generator
    PARAM_SCALE = 50.0  # Scale factor for parameter space (font units)

    # GIF settings
    GIF_SAMPLE_RATE = 10

    # Parse font path
    if len(sys.argv) > 1:
        ttf_path = sys.argv[1]
    else:
        ttf_path = find_system_font()

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    font_name = Path(ttf_path).stem.replace(" ", "_")
    experiment_name = f"{font_name}_{CHAR}_{timestamp}"
    experiment_dir = Path(__file__).parent.parent / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Output dir: {experiment_dir}")
    print(f"Font: {ttf_path}")
    print(f"Character: '{CHAR}'")
    print(f"Resolution: {WIDTH}x{HEIGHT}")

    # Create target image (unmodified glyph)
    print("\nCreating target image...")
    commands = load_glyph_commands(ttf_path, CHAR)
    topology = build_topology(commands)
    V_zero = torch.zeros(2 * topology.base_points.shape[0])
    base_commands = apply_vector(topology, V_zero)
    target_np = rasterize(base_commands, WIDTH, HEIGHT)

    # Create environment
    env = GlyphEnv(
        ttf_path=ttf_path,
        char=CHAR,
        target_image_np=target_np,
        width=WIDTH,
        height=HEIGHT,
    )

    F_DIM = env.dim
    print(f"Parameter dimension: {F_DIM}")
    print(f"Number of control points: {env.num_points}")

    # Create function wrapper
    test_function = GlyphFitFunction(env, scale=PARAM_SCALE)

    # Setup gfog components
    DEVICE = torch.device("cpu")

    fn = components.Fn(
        f=test_function,
        input_dim=F_DIM,
        device=DEVICE,
        dtype=torch.float,
    )

    # Generator and Discriminator networks
    # Generator: latent -> parameter space
    # Use deeper network for higher dimensional problem
    G = MLP(
        input_dim=LATENT_DIM,
        output_dim=F_DIM,
        hidden_dims=[64, 64],
    ).to(DEVICE)

    # Discriminator: parameter space -> scalar
    D = MLP(
        input_dim=F_DIM,
        output_dim=1,
        hidden_dims=[64, 64],
    ).to(DEVICE)

    # Buffer to store elite samples
    buffer = components.BufferComp(B=Buffer(buffer_size=2 * BATCH_SIZE))

    # GAN components
    gan = components.GAN(
        G=G,
        D=D,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=WangIsolaUniformity(
            config=WangIsolaUniformityConfig(use_buffer=True, weight=50),
            buffer=buffer.B,
        ),
        latent_dim=LATENT_DIM,
        optimizerG=torch.optim.Adam(lr=0.005, params=G.parameters()),
        optimizerD=torch.optim.Adam(lr=0.05, params=D.parameters()),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d), b=BATCH_SIZE, d=LATENT_DIM
        ),
        device=DEVICE,
        dtype=torch.float,
    )

    # Optimizer
    optimizer = DefaultOpt(
        components.OptComponents(fn=fn, gan=gan, batch_size=BATCH_SIZE, buffer=buffer)
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
    print(f"\nStarting optimization for {N_ITER} iterations...")
    best_values = []
    buffer_history = []
    iteration_numbers = []

    with progress:
        task = progress.add_task("Optimizing", total=N_ITER, best=999.0, mean=999.0)

        for i in range(N_ITER):
            optimizer.step()

            best_val = buffer.B.get_value(0)
            mean_val = buffer.B.get_mean_buffer_value()
            best_values.append(best_val)

            if i % GIF_SAMPLE_RATE == 0 or i == N_ITER - 1:
                buffer_history.append(buffer.B.tensor_buffer.clone())
                iteration_numbers.append(i)

            progress.update(task, advance=1, best=best_val, mean=mean_val)

    # Results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)

    best_params = buffer.B.get_top_k(1).squeeze()
    best_loss = buffer.B.get_value(0)

    print(f"Best loss: {best_loss:.6f}")
    print(f"Final mean buffer loss: {buffer.B.get_mean_buffer_value():.6f}")

    # Save visualizations
    try:
        from PIL import Image

        # Save target
        target_img = Image.fromarray((target_np * 255).astype(np.uint8))
        target_img.save(experiment_dir / "target.png")
        print(f"\nSaved target to {experiment_dir}/target.png")

        # Save best result
        # Scale back from normalized to font units
        best_V = best_params * PARAM_SCALE
        best_img_np = env.render(best_V)
        best_img = Image.fromarray((best_img_np * 255).astype(np.uint8))
        best_img.save(experiment_dir / "best.png")
        print(f"Saved best result to {experiment_dir}/best.png")

        # Save comparison (target | best side by side)
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
            ax.set_title(f"Glyph Fitting Optimization - Character '{CHAR}'")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            fig.savefig(experiment_dir / "loss_curve.png", dpi=150)
            plt.close(fig)
            print(f"Saved loss curve to {experiment_dir}/loss_curve.png")
        except ImportError:
            print("Note: Install matplotlib to save loss curve")

        # Save experiment config
        config = {
            "font": ttf_path,
            "char": CHAR,
            "width": WIDTH,
            "height": HEIGHT,
            "n_iter": N_ITER,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "param_scale": PARAM_SCALE,
            "best_loss": best_loss,
            "final_mean_loss": buffer.B.get_mean_buffer_value(),
            "param_dim": F_DIM,
            "num_points": env.num_points,
        }
        import json

        with open(experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {experiment_dir}/config.json")

        # Save best parameters
        np.save(experiment_dir / "best_params.npy", best_V.detach().cpu().numpy())
        print(f"Saved best parameters to {experiment_dir}/best_params.npy")

    except ImportError:
        print("\nNote: Install pillow to save visualization images")


if __name__ == "__main__":
    main()
