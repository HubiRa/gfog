import torch
import numpy as np
from typing import List
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import imageio.v3 as iio
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")


def plot_optimization_gif(
    test_function,
    x_range: torch.Tensor,
    y_range: torch.Tensor,
    buffer_history: List[torch.Tensor],
    iteration_numbers: List[int],
    filename: str = "optimization.gif",
    cmap: str = "crest",
    n_contours: int = 30,
    figsize: tuple = (10, 8),
    show_minima: bool = True,
    minima_alpha: float = 0.3,
    fps: int = 10,
    max_points: int = 100,
    dpi: int = 80,
    quantize_colors: bool = True,
) -> None:
    if test_function.input_dim != 2:
        raise ValueError("GIF plotting only supported for 2D functions")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
    grid_points = torch.stack([X, Y], dim=-1)
    Z = test_function.f(grid_points)

    # Create better normalization for contour colors
    z_min, z_max = Z.min().item(), Z.max().item()
    z_range = z_max - z_min
    if z_range < 1e-6:  # Very small range, use symmetric normalization
        z_center = (z_min + z_max) / 2
        z_extent = max(abs(z_min - z_center), abs(z_max - z_center), 1e-6)
        norm = Normalize(vmin=z_center - z_extent, vmax=z_center + z_extent)
    else:
        # Use percentile-based normalization to enhance contrast
        z_flat = Z.flatten()
        z_5th = torch.quantile(z_flat, 0.05).item()
        z_95th = torch.quantile(z_flat, 0.95).item()
        norm = Normalize(vmin=z_5th, vmax=z_95th)

    logger.info(f"Generating {len(buffer_history)} frames for GIF...")

    frames = []
    for i, (buffer_points, iteration) in tqdm(
        enumerate(zip(buffer_history, iteration_numbers))
    ):
        # if i % 10 == 0:
        #     logger.info(f"Generating frame {i + 1}/{len(buffer_history)}")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Plot contours
        contour_filled = ax.contourf(
            X,
            Y,
            Z,
            levels=n_contours,
            cmap=sns.color_palette(cmap, as_cmap=True),
            alpha=0.85,
            norm=norm,
        )
        ax.contour(
            X, Y, Z, levels=n_contours, colors="white", alpha=0.4, linewidths=1.0
        )

        # Plot known minima
        if (
            show_minima
            and hasattr(test_function, "known_minima")
            and test_function.known_minima is not None
        ):
            minima_color = sns.color_palette("Set2")[4]  # Nice purple
            for minimum in test_function.known_minima:
                min_np = minimum.detach().cpu().numpy()
                ax.plot(
                    min_np[0],
                    min_np[1],
                    "*",
                    color=minima_color,
                    markersize=22,
                    markeredgecolor="white",
                    markeredgewidth=2.5,
                    alpha=minima_alpha,
                    zorder=7,
                )

        # Plot buffer points
        if len(buffer_points) > 0:
            n_show = min(max_points, len(buffer_points))
            points_to_show = buffer_points[:n_show]
            points_np = points_to_show.detach().cpu().numpy()

            buffer_color = sns.color_palette("Set2")[1]
            ax.scatter(
                points_np[:, 0],
                points_np[:, 1],
                color=buffer_color,
                s=40,
                alpha=0.85,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
            )

        # Setup axis
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(y_range[0], y_range[-1])
        ax.set_xlabel("X", fontsize=13, fontweight="bold")
        ax.set_ylabel("Y", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{test_function.__class__.__name__} - Iteration {iteration}",
            fontsize=16,
            fontweight="bold",
        )

        # Save frame to memory
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frame = frame[:, :, :3]  # Remove alpha channel, keep RGB
        frames.append(frame)

        plt.close(fig)

    # Create GIF using imageio
    logger.info(f"Saving GIF to {output_path}...")

    # Apply color quantization if requested
    if quantize_colors:
        iio.imwrite(
            output_path,
            frames,
            duration=250 // fps,  # duration in milliseconds
            loop=0,  # infinite loop
            quantizer="nq",  # Neural quantizer for better quality
            palettesize=256,  # Standard GIF palette size
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            duration=250 // fps,  # duration in milliseconds
            loop=0,  # infinite loop
        )

    logger.info("GIF saved successfully!")
