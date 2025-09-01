from gfog.curiosity import (
    WangIsolaUniformityConfig,
    WangIsolaUniformity,
)
import torch

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from gfog.opt import DefaultOpt
from gfog.models import MLP
from gfog.opt import components
from gfog.buffer import Buffer
from gfog.opt.latents_sampler import LatentSamplerLambda
from torch.nn import BCEWithLogitsLoss
from loguru import logger

from functions import (
    HimmelblauFunction,
)
from plotting import plot_optimization_gif
from pathlib import Path

# --------------------
# Function to optimize
# --------------------

TEST_FUNCTION = HimmelblauFunction()
F_DIM = TEST_FUNCTION.input_dim

fn = components.Fn(
    f=TEST_FUNCTION,
    input_dim=F_DIM,
    device=torch.device("cpu"),
    dtype=torch.float,
)
plot_ranges = TEST_FUNCTION.get_plot_ranges(n_points=100)


# -------------------------
# GAN used for optimization
# -------------------------

LATENT_DIM = 10
BATCH_SIZE = 64

GAN_DEVICE = torch.device("cpu")
G = MLP(input_dim=LATENT_DIM, output_dim=F_DIM, hidden_dims=[32]).to(GAN_DEVICE)
D = MLP(input_dim=F_DIM, output_dim=1, hidden_dims=[32]).to(GAN_DEVICE)


buffer = components.BufferComp(B=Buffer(buffer_size=2 * BATCH_SIZE))

logger.debug(f"{buffer.B.get_sorted_values()}")

gan = components.GAN(
    G=G,
    D=D,
    loss=BCEWithLogitsLoss(),
    curiosity_loss=WangIsolaUniformity(
        config=WangIsolaUniformityConfig(use_buffer=True, weight=100),
        buffer=buffer.B,
    ),
    latent_dim=LATENT_DIM,
    optimizerG=torch.optim.Adam(lr=0.01, params=G.parameters()),
    optimizerD=torch.optim.Adam(lr=0.1, params=D.parameters()),
    latent_sampler=LatentSamplerLambda(
        lambda b, d: torch.randn(b, d) * 4, b=BATCH_SIZE, d=LATENT_DIM
    ),
    device=GAN_DEVICE,
    dtype=torch.float,
)

# -------------------------------
# Compose optimization components
# -------------------------------

optimizer = DefaultOpt(
    components.OptComponents(fn=fn, gan=gan, batch_size=BATCH_SIZE, buffer=buffer)
)

progress = Progress(
    TextColumn("Iteration {task.completed}"),
    BarColumn(),
    TextColumn("Best: {task.fields[best]:.4f}"),
    TextColumn("Mean: {task.fields[mean]:.4f}"),
    TimeElapsedColumn(),
)


def main() -> None:
    N_ITER = 1000
    GIF_SAMPLE_RATE = 5

    buffer_history = []
    iteration_numbers = []
    best_values = []

    logger.info("Staring optimization")
    with progress:
        task = progress.add_task("Optimizing", total=N_ITER, best=999.0, mean=999.0)
        for i in range(N_ITER):
            optimizer.step()

            if i % GIF_SAMPLE_RATE == 0 or i == N_ITER - 1:
                buffer_history.append(buffer.B.tensor_buffer.clone())
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

    logger.info(f"Creating GIF with {len(buffer_history)} frames")
    plot_optimization_gif(
        test_function=TEST_FUNCTION,
        x_range=plot_ranges.x,
        y_range=plot_ranges.y,
        buffer_history=buffer_history,
        iteration_numbers=iteration_numbers,
        filename=str(
            (
                Path(__file__).parent / "../../assets/himmelblau_optimization.gif"
            ).resolve()
        ),
        fps=10,
        max_points=buffer.B.buffer_size,
        minima_alpha=0.5,
    )


if __name__ == "__main__":
    main()
