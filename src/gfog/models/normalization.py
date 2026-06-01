from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


OutputNormMode = Literal["none", "l2", "centered_l2", "layernorm"]


def normalize_output(
    x: torch.Tensor,
    mode: OutputNormMode,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize generated candidate tensors in representation space."""
    if mode == "none":
        return x
    if mode == "l2":
        return F.normalize(x, p=2, dim=dim, eps=eps)
    if mode == "centered_l2":
        centered = x - x.mean(dim=dim, keepdim=True)
        return F.normalize(centered, p=2, dim=dim, eps=eps)
    if mode == "layernorm":
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mean) / std
    raise ValueError(f"Unknown output normalization mode: {mode}")


class OutputNormalizer(nn.Module):
    """Wrap a module and normalize its output per sample."""

    def __init__(
        self,
        module: nn.Module,
        mode: OutputNormMode = "none",
        *,
        dim: int = -1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.module = module
        self.mode = mode
        self.dim = dim
        self.eps = eps

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = self.module(*args, **kwargs)
        return normalize_output(x, self.mode, dim=self.dim, eps=self.eps)
