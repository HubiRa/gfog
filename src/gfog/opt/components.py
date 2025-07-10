from dataclasses import dataclass
from typing import Callable, Tuple
import torch
from torch import nn
from ..curiosity import CuriosityLossBase
from ..buffer.base import BufferBase
from .latents_sampler import LatentSamplerBase


@dataclass
class Fn:
    f: Callable[[torch.Tensor], torch.Tensor]
    input_dim: int
    device: torch.device
    dtype: torch.dtype


LatentDim = int | Tuple[int, int]


@dataclass
class GAN:
    G: nn.Module
    D: nn.Module
    loss: nn.Module
    curiosity_loss: CuriosityLossBase
    latent_dim: LatentDim
    optimizerG: torch.optim.Optimizer
    optimizerD: torch.optim.Optimizer
    latent_sampler: LatentSamplerBase
    device: torch.device
    dtype: torch.dtype


@dataclass
class Buffer:
    B: BufferBase
    # TODO elite selection


@dataclass
class OptComponents:
    fn: Fn
    gan: GAN
    batch_size: int
    buffer: Buffer
