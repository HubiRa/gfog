from dataclasses import dataclass
from typing import Any, Callable, Literal, Tuple

import torch
from torch import nn

from ..buffer import Buffer
from ..curiosity.curiosity import CuriosityLossBase
from .latents_sampler import LatentSamplerBase


@dataclass
class Fn:
    f: Callable[[torch.Tensor], Any]
    input_dim: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")


LatentDim = int | Tuple[int, int]
EliteSamplingMode = Literal["top_k", "random_top_k"]


@dataclass
class GAN:
    G: nn.Module
    D: nn.Module
    loss: nn.Module
    curiosity_loss: CuriosityLossBase | None
    latent_dim: LatentDim
    optimizerG: torch.optim.Optimizer
    optimizerD: torch.optim.Optimizer
    latent_sampler: LatentSamplerBase
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if isinstance(self.latent_dim, int):
            if self.latent_dim <= 0:
                raise ValueError(f"latent_dim must be > 0, got {self.latent_dim}")
        else:
            if len(self.latent_dim) != 2 or any(dim <= 0 for dim in self.latent_dim):
                raise ValueError(
                    f"latent_dim tuple must contain two positive ints, got {self.latent_dim}"
                )


@dataclass
class BufferComp:
    B: Buffer

    def __post_init__(self) -> None:
        if self.B.buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got {self.B.buffer_size}")


@dataclass
class OptComponents:
    fn: Fn
    gan: GAN
    batch_size: int
    buffer: BufferComp
    discriminator_steps: int = 1
    elite_sampling: EliteSamplingMode = "random_top_k"
    elite_pool_size: int | None = None
    weight_clip: float | None = None
    gradient_penalty_weight: float = 10.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.discriminator_steps <= 0:
            raise ValueError(
                f"discriminator_steps must be > 0, got {self.discriminator_steps}"
            )
        if self.elite_sampling not in {"top_k", "random_top_k"}:
            raise ValueError("elite_sampling must be one of {'top_k', 'random_top_k'}")
        if self.elite_pool_size is not None and self.elite_pool_size <= 0:
            raise ValueError(
                f"elite_pool_size must be > 0 when set, got {self.elite_pool_size}"
            )
        if self.weight_clip is not None and self.weight_clip <= 0:
            raise ValueError(
                f"weight_clip must be > 0 when set, got {self.weight_clip}"
            )
        if self.gradient_penalty_weight <= 0:
            raise ValueError(
                "gradient_penalty_weight must be > 0, got "
                f"{self.gradient_penalty_weight}"
            )
