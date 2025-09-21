import gymnasium as gym
import numpy as np
from gfog.curiosity import (
    WangIsolaUniformityConfig,
    WangIsolaUniformity,
)
from gfog.opt import DefaultOpt
from gfog.models import MLP
from gfog.opt import components
from gfog.buffer import Buffer, Levels
from gfog.opt.latents_sampler import LatentSamplerLambda
import torch
from torch.nn import BCEWithLogitsLoss
from loguru import logger


# --------------------------------------
# Cart pole example
# --------------------------------------
N_EPISODE_STEPS = 500
env = gym.make("CartPole-v1", max_episode_steps=N_EPISODE_STEPS)


class CartPole:
    def __init__(self, env, runs_per_env: int = 5):
        self.env = env
        self.runs_per_env = runs_per_env

    @staticmethod
    def run_env(t, env):
        obs, _ = env.reset(seed=None)
        total = 0
        done = False
        while not done:
            act = 0 if (obs @ t[:4] + t[4]) < 0 else 1
            obs, r, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            total += r
        return total

    def __call__(self, theta):
        if isinstance(theta, torch.Tensor):
            theta = theta.cpu().numpy()
        # tiny linear policy over obs -> action

        returns = []
        for t in theta:
            runs = []
            for _ in range(self.runs_per_env):
                total = CartPole.run_env(t, self.env)
                runs.append(total)
            # returns.append(-np.mean(runs))
            returns.append(-np.median(runs))
        return returns  # minimize


# --------------------------------------
# Setting up Optimizer components
# --------------------------------------

F_DIM = 5
LATENT_DIM = 10
BATCH_SIZE = 16

fn = components.Fn(
    f=CartPole(env),
    input_dim=F_DIM,
    device=torch.device("cpu"),
    dtype=torch.float,
)

GAN_DEVICE = torch.device("cpu")
G = MLP(input_dim=LATENT_DIM, output_dim=F_DIM, hidden_dims=[32]).to(GAN_DEVICE)
D = MLP(input_dim=F_DIM, output_dim=1, hidden_dims=[32], use_spectral_norm=False).to(
    GAN_DEVICE
)

buffer = components.BufferComp(
    B=Buffer(buffer_size=2 * BATCH_SIZE, value_levels=Levels(["median return"]))
)

gan = components.GAN(
    G=G,
    D=D,
    loss=BCEWithLogitsLoss(),
    curiosity_loss=WangIsolaUniformity(
        config=WangIsolaUniformityConfig(use_buffer=True, weight=10),
        buffer=buffer.B,
    ),
    latent_dim=LATENT_DIM,
    optimizerG=torch.optim.Adam(lr=0.01, params=G.parameters()),
    optimizerD=torch.optim.Adam(lr=0.1, params=D.parameters()),
    latent_sampler=LatentSamplerLambda(
        lambda b, d: torch.randn(b, d) * 2, b=BATCH_SIZE, d=LATENT_DIM
    ),
    device=GAN_DEVICE,
    dtype=torch.float,
)

# --------------------------------------
# Optimize
# --------------------------------------

optimizer = DefaultOpt(
    components.OptComponents(fn=fn, gan=gan, batch_size=BATCH_SIZE, buffer=buffer)
)
logger.info(
    f"Performance of best 5 samples in buffer after random init (optimum: -{N_EPISODE_STEPS})"
)
buffer.B.print_values(slice(0, 5, 1))


optimizer.optimize(n_iter=200, verbous=True)
logger.info(
    f"Performance of best 5 samples in buffer after optimization (optimum: -{N_EPISODE_STEPS}))"
)
buffer.B.print_values(slice(0, 5, 1))

# run env for longer to see if method generalizes to more time steps
N_EPISODE_STEPS = 5000
env2 = gym.make("CartPole-v1", max_episode_steps=N_EPISODE_STEPS)
cart_pole = CartPole(env2, runs_per_env=20)
results = cart_pole(buffer.B.get_top_k(5))
logger.info("Teesting generalizaton:")
logger.info(f"  ==> Result for {N_EPISODE_STEPS = } (top 5): {results = }")
