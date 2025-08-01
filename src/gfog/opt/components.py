from dataclasses import dataclass
from typing import Callable, Tuple, Any
import torch
from torch import nn
from ..curiosity.curiosity import CuriosityLossBase

# from ..buffer.base import BufferBase, Buffer
from ..buffer import Buffer
from .latents_sampler import LatentSamplerBase


@dataclass
class Fn:
    f: Callable[[torch.Tensor], Any]
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
class BufferComp:
    B: Buffer
    # TODO: elite selection


@dataclass
class OptComponents:
    fn: Fn
    gan: GAN
    batch_size: int
    buffer: BufferComp
