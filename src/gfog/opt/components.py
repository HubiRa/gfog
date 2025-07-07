from dataclasses import dataclass
from typing import Callable, TypeAlias, Tuple
import torch
from torch import nn


@dataclass
class Fn:
    f: Callable[[torch.Tensor], torch.Tensor]
    input_dim: int
    device: torch.device
    dtype: torch.dtype


LatentDim = int | Tuple[int, int]


@dataclass
class Models:
    G: nn.Module
    D: nn.Module
    latent_dim: LatentDim
    batch_size: int
    optimizerG: torch.optim.Optimizer
    optimizerD: torch.optim.Optimizer
    device: torch.device
    dtype: torch.dtype


@dataclass
class OptConfig:
    f: Fn
    models: Models
