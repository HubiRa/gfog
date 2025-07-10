from gfog.curiosity import (
    CuriositySiglipLoss,
    CuriositySiglipLossConfig,
)
import torch

from gfog.opt import DefaultOpt
from gfog.models import MLP
from gfog.opt import components
from gfog.buffer.default import SimpleBuffer
from gfog.opt.components import GAN
from gfog.opt.components import Fn
from gfog.opt.latents_sampler import LatentSamplerLambda
from torch.nn import BCEWithLogitsLoss
from loguru import logger

from functions import (
    HimmelblauFunction,
)
from plotting import plot_contour, plot_optimization_gif


# --------------------
# Function to optimize
# --------------------

TEST_FUNCTION = HimmelblauFunction()
F_DIM = TEST_FUNCTION.input_dim  # Fixed: removed () - it's a property not method

fn = Fn(
    f=TEST_FUNCTION,
    input_dim=F_DIM,
    device=torch.device("cpu"),
    dtype=torch.float,
)

# Generate plot ranges from test function domain
plot_ranges = TEST_FUNCTION.get_plot_ranges(n_points=100)


# -------------------------
# GAN used for optimization
# -------------------------

LATENT_DIM = 10
BATCH_SIZE = 64

GAN_DEVICE = torch.device("cpu")
G = MLP(input_dim=LATENT_DIM, output_dim=F_DIM, hidden_dims=[256]).to(GAN_DEVICE)
D = MLP(input_dim=F_DIM, output_dim=1, hidden_dims=[256], spectral_norm=False).to(
    GAN_DEVICE
)


buffer = components.Buffer(B=SimpleBuffer(buffer_size=2 * BATCH_SIZE))

gan = GAN(
    G=G,
    D=D,
    loss=BCEWithLogitsLoss(),
    curiosity_loss=CuriositySiglipLoss(
        config=CuriositySiglipLossConfig(calc_self_sim=100.0, calc_cross_sim=100.0),
        buffer=buffer.B,
    ),
    latent_dim=LATENT_DIM,
    optimizerG=torch.optim.Adam(lr=0.01, params=G.parameters()),
    optimizerD=torch.optim.Adam(lr=0.1, params=D.parameters()),
    latent_sampler=LatentSamplerLambda(lambda: torch.rand(BATCH_SIZE, LATENT_DIM)),
    device=GAN_DEVICE,
    dtype=torch.float,
)

# -------------------------------
# Compose optimization components
# -------------------------------

optimizer = DefaultOpt(
    components.OptComponents(fn=fn, gan=gan, batch_size=BATCH_SIZE, buffer=buffer)
)


def main() -> None:
    N_ITER = 1000  # Reduced for testing
    N_LOG = 100
    GIF_SAMPLE_RATE = 20

    buffer_history = []
    iteration_numbers = []
    best_values = []

    for i in range(N_ITER):
        optimizer.step()

        if i % GIF_SAMPLE_RATE == 0 or i == N_ITER - 1:  # Include first and last
            all_buffer_points = buffer.B.get_top_k(k=buffer.B.buffer_size)
            buffer_history.append(all_buffer_points.clone())
            iteration_numbers.append(i)  # Track the true iteration number

        best_values.append(buffer.B.values[0])
        if i % N_LOG == 0:
            print(f"Iteration {i}: best value {best_values[-1]}")

    logger.info(
        f"found minimum at {buffer.B.get_top_k(1)} with value {buffer.B.values[0]}"
    )

    logger.info(f"Creating GIF with {len(buffer_history)} frames")
    logger.info("Creating optimization animation...")
    plot_optimization_gif(
        test_function=TEST_FUNCTION,
        x_range=plot_ranges.x,
        y_range=plot_ranges.y,
        buffer_history=buffer_history,
        iteration_numbers=iteration_numbers,
        filename="himmelblau_optimization.gif",
        fps=3,
        max_points=buffer.B.buffer_size,
        minima_alpha=0.2,
    )

    logger.info("Plotting final buffer state with ALL buffer points...")
    all_buffer_points = buffer.B.get_top_k(k=buffer.B.buffer_size).to("cpu")
    plot_contour(
        test_function=TEST_FUNCTION,
        x_range=plot_ranges.x,
        y_range=plot_ranges.y,
        final_points=all_buffer_points,
        minima_alpha=0.2,  # Make minima very transparent
    )


if __name__ == "__main__":
    main()
