import torch
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from loguru import logger

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.style.use("seaborn-v0_8-darkgrid")


def plot_contour(
    test_function,
    x_range: torch.Tensor,
    y_range: torch.Tensor,
    optimization_path: Optional[torch.Tensor] = None,
    final_points: Optional[torch.Tensor] = None,
    cmap: str = "crest",
    time_cmap: str = "plasma",
    n_contours: int = 50,
    figsize: tuple = (10, 8),
    show_minima: bool = True,
    minima_alpha: float = 0.3,
) -> None:
    if test_function.input_dim != 2:
        raise ValueError("Contour plotting only supported for 2D functions")

    X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
    grid_points = torch.stack([X, Y], dim=-1)
    Z = test_function.f(grid_points)

    fig, ax = plt.subplots(figsize=figsize)
    contour_filled = ax.contourf(
        X,
        Y,
        Z,
        levels=n_contours,
        cmap=sns.color_palette(cmap, as_cmap=True),
        alpha=0.85,
    )
    cbar_contour = plt.colorbar(contour_filled, ax=ax, shrink=0.8, pad=0.02)
    cbar_contour.set_label("Function Value", rotation=270, labelpad=15, fontsize=11)

    # Plot contour lines with seaborn styling
    contour_lines = ax.contour(
        X, Y, Z, levels=n_contours, colors="white", alpha=0.4, linewidths=1.0
    )

    # Plot optimization path with time-based color gradient
    if optimization_path is not None:
        path_np = optimization_path.detach().cpu().numpy()
        n_points = len(path_np)

        # Create time values (0 to 1) for color mapping
        time_values = np.linspace(0, 1, n_points)

        # Plot all points with time-based colors using seaborn palette
        scatter = ax.scatter(
            path_np[:, 0],
            path_np[:, 1],
            c=time_values,
            cmap=sns.color_palette(time_cmap, as_cmap=True),
            s=60,
            alpha=0.9,
            zorder=5,
            edgecolors="white",
            linewidths=1.2,
        )

        # Add colorbar for time gradient
        cbar_time = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.12)
        cbar_time.set_label("Time Progress", rotation=270, labelpad=15, fontsize=11)

        # Mark start and end points with seaborn colors
        start_color = sns.color_palette("Set2")[2]  # Nice green
        end_color = sns.color_palette("Set2")[3]  # Nice red
        ax.plot(
            path_np[0, 0],
            path_np[0, 1],
            "o",
            color=start_color,
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Start",
            zorder=6,
        )
        ax.plot(
            path_np[-1, 0],
            path_np[-1, 1],
            "s",
            color=end_color,
            markersize=14,
            markeredgecolor="white",
            markeredgewidth=2,
            label="End",
            zorder=6,
        )

    # Plot final points if provided
    if final_points is not None:
        final_np = final_points.detach().cpu().numpy()
        buffer_color = sns.color_palette("Set2")[1]  # Nice orange
        ax.scatter(
            final_np[:, 0],
            final_np[:, 1],
            color=buffer_color,
            s=45,
            marker="o",
            alpha=0.8,
            label="Buffer Points",
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )

    # Plot known minima if available (with transparency)
    if (
        show_minima
        and hasattr(test_function, "known_minima")
        and test_function.known_minima is not None
    ):
        minima_color = sns.color_palette("Set2")[4]  # Nice purple
        for i, minimum in enumerate(test_function.known_minima):
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
                label="Global Minimum" if i == 0 else "",
                zorder=7,
            )

    ax.set_xlabel("X", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{test_function.__class__.__name__} Contour Plot",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    # Seaborn already provides nice grid, just adjust it
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


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
) -> None:
    if test_function.input_dim != 2:
        raise ValueError("GIF plotting only supported for 2D functions")

    X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
    grid_points = torch.stack([X, Y], dim=-1)
    Z = test_function.f(grid_points)

    fig, ax = plt.subplots(figsize=figsize)
    contour_filled = ax.contourf(
        X,
        Y,
        Z,
        levels=n_contours,
        cmap=sns.color_palette(cmap, as_cmap=True),
        alpha=0.85,
    )
    ax.contour(X, Y, Z, levels=n_contours, colors="white", alpha=0.4, linewidths=1.0)

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

    # Initialize empty scatter plot for buffer points
    # All points at same time have same color (no color gradient within frame)
    buffer_color = sns.color_palette("Set2")[1]
    scat = ax.scatter(
        [],
        [],
        color=buffer_color,
        s=40,
        alpha=0.85,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
    )

    # Setup axis with seaborn styling
    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_ylim(y_range[0], y_range[-1])
    ax.set_xlabel("X", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    title = ax.set_title(f"{test_function.__class__.__name__} - Iteration 0")

    def animate(frame):
        if frame < len(buffer_history):
            # Get all buffer points for this iteration
            buffer_points = buffer_history[frame]

            if len(buffer_points) > 0:
                # Limit number of points to show for clarity
                n_show = min(max_points, len(buffer_points))
                points_to_show = buffer_points[:n_show]

                # Convert to numpy
                points_np = points_to_show.detach().cpu().numpy()

                # All points have consistent color - no need for time encoding since GIF shows time
                # Update scatter plot with points positions only
                scat.set_offsets(points_np)

                # Update title with true iteration number and seaborn styling
                true_iteration = iteration_numbers[frame]
                title.set_text(
                    f"{test_function.__class__.__name__} - Iteration {true_iteration}"
                )
                title.set_fontsize(16)
                title.set_fontweight("bold")

        return scat, title

    # Create animation
    n_frames = len(buffer_history)
    logger.info(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig, animate, frames=n_frames, interval=1000 // fps, blit=False, repeat=True
    )

    # Save as GIF with optimization
    logger.info(f"Saving optimization animation to {filename}...")
    writer = PillowWriter(fps=fps, bitrate=1800)
    anim.save(
        filename,
        writer=writer,
        progress_callback=lambda i, n: print(f"Frame {i + 1}/{n}")
        if i % 10 == 0
        else None,
    )
    logger.info("Animation saved successfully!")

    plt.close(fig)
