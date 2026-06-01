from gfog.curiosity import (
    WangIsolaUniformityConfig,
    WangIsolaUniformity,
)
import torch

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from gfog.opt import DeltaOpt
from gfog.models import MLP
from gfog.opt import components
from gfog.buffer import Buffer
from gfog.opt.latents_sampler import LatentSamplerLambda
from torch.nn import BCEWithLogitsLoss
from loguru import logger

from functions import HimmelblauFunction
from plotting import plot_optimization_gif
from pathlib import Path


def main() -> None:
    # --------------------
    # Function to optimize
    # --------------------
    test_fn = HimmelblauFunction()
    f_dim = test_fn.input_dim

    fn = components.Fn(
        f=test_fn,
        input_dim=f_dim,
        device=torch.device("cpu"),
        dtype=torch.float,
    )

    # -------------------------
    # GAN used for optimization
    # -------------------------
    latent_dim = 10
    batch_size = 8

    gan_device = torch.device("cpu")
    G = MLP(input_dim=latent_dim, output_dim=f_dim, hidden_dims=[32]).to(gan_device)
    D = MLP(input_dim=f_dim, output_dim=1, hidden_dims=[32]).to(gan_device)

    buffer = components.BufferComp(B=Buffer(buffer_size=2 * batch_size))

    gan = components.GAN(
        G=G,
        D=D,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=WangIsolaUniformity(
            config=WangIsolaUniformityConfig(use_buffer=True, weight=1),
            buffer=buffer.B,
        ),
        latent_dim=latent_dim,
        optimizerG=torch.optim.Adam(lr=0.01, params=G.parameters()),
        optimizerD=torch.optim.Adam(lr=0.1, params=D.parameters()),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d) * 4, b=batch_size, d=latent_dim
        ),
        device=gan_device,
        dtype=torch.float,
    )

    # -------------------------------
    # Compose optimization components
    # -------------------------------
    opt = DeltaOpt(
        components.OptComponents(fn=fn, gan=gan, batch_size=batch_size, buffer=buffer)
    )

    # Initialize departure points (multi-start). If None, random points are used.
    n_departure_points = 1
    opt.init_departure_points(dep_points=None, n_points=n_departure_points)

    # --------------------
    # Plot/GIF preparation
    # --------------------
    plot_ranges = test_fn.get_plot_ranges(n_points=100)

    # ----------------
    # Run optimization
    # ----------------
    n_iter = 100
    gif_sample_rate = 5
    buffer_history = []
    iteration_numbers = []
    best_values = []

    logger.info("Starting DeltaOpt on Himmelblau...")

    progress = Progress(
        TextColumn("Iteration {task.completed}"),
        BarColumn(),
        TextColumn("Best: {task.fields[best]:.4f}"),
        TextColumn("Mean: {task.fields[mean]:.4f}"),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task("Optimizing", total=n_iter, best=999.0, mean=999.0)
        for i in range(n_iter):
            opt.step()

            if i % gif_sample_rate == 0 or i == n_iter - 1:
                buffer_history.append(opt.current)
                iteration_numbers.append(i)

            best_values.append(buffer.B.get_value(0))
            progress.update(
                task,
                advance=1,
                best=best_values[-1],
                mean=buffer.B.get_mean_buffer_value(),
            )

    logger.info(f"Best point found: {buffer.B.get_top_k(1).squeeze().tolist()}")
    logger.info(f"Best value: {buffer.B.get_value(0):.6f}")

    # ----------------
    # Create GIF output
    # ----------------
    logger.info(f"Creating GIF with {len(buffer_history)} frames")
    plot_optimization_gif(
        test_function=test_fn,
        x_range=plot_ranges.x,
        y_range=plot_ranges.y,
        buffer_history=buffer_history,
        iteration_numbers=iteration_numbers,
        filename=str(
            (
                Path(__file__).parent / "../../assets/himmelblau_delta_optimization.gif"
            ).resolve()
        ),
        fps=10,
        max_points=buffer.B.buffer_size,
        minima_alpha=0.5,
    )


if __name__ == "__main__":
    main()
