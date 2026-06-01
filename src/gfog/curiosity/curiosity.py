from dataclasses import dataclass
import warnings

import torch
from torch import nn

from ..buffer import Buffer
from ..utils import (
    cross_siglip,
    cross_similarity_loss,
    self_siglip,
    self_similarity_loss,
    uniformity_loss,
)
from .scheduler import Scheduler


@dataclass
class CuriosityLossBaseConfig:
    pass


class CuriosityLossBase(nn.Module):
    def __init__(
        self,
        config: CuriosityLossBaseConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.buffer = buffer
        self.scheduler = scheduler

    def _buffer_top_k_like(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError("Buffer-dependent curiosity loss requires a buffer")
        k = min(k, len(self.buffer))
        return self.buffer.get_top_k(k).to(device=x.device, dtype=x.dtype)


@dataclass
class CuriosityLossConfig(CuriosityLossBaseConfig):
    temperature: float = 0.7
    # Values <= 0 explicitly disable the corresponding term.
    calc_self_sim: float = 0.5
    calc_cross_sim: float = 0.5

    def __post_init__(self) -> None:
        self.calc_cross_sim = self.calc_cross_sim if self.calc_cross_sim > 0.0 else 0.0
        self.calc_self_sim = self.calc_self_sim if self.calc_self_sim > 0.0 else 0.0


class CuriosityLoss(CuriosityLossBase):
    """Repulsion-style curiosity loss based on cosine similarity logits."""

    def __init__(
        self,
        config: CuriosityLossConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, buffer, scheduler)
        self.config = config
        warnings.warn(
            "CuriosityLoss is a legacy curiosity objective. Prefer WangIsolaUniformity for new code.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.config.calc_cross_sim and buffer is None:
            raise ValueError(
                "Cross similarity loss requires a buffer because it compares generator outputs against elite buffer samples"
            )

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        batch_size = g_out.size(0)
        loss = torch.zeros((), device=g_out.device, dtype=g_out.dtype)
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.calc_cross_sim and self.buffer is not None:
            elite = self._buffer_top_k_like(g_out, batch_size)
            loss = (
                loss
                + sched_value
                * self.config.calc_cross_sim
                * cross_similarity_loss(
                    g_out,
                    elite,
                    temperature=self.config.temperature,
                )
            )

        if self.config.calc_self_sim:
            loss = (
                loss
                + sched_value
                * self.config.calc_self_sim
                * self_similarity_loss(g_out, temperature=self.config.temperature)
            )
        return loss


@dataclass
class CuriositySiglipLossConfig(CuriosityLossConfig):
    temperature: float = 1.0


class CuriositySiglipLoss(CuriosityLoss):
    """Repulsion-style curiosity loss using SigLIP-style negative pairs only."""

    def __init__(
        self,
        config: CuriositySiglipLossConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, buffer, scheduler, **kwargs)
        warnings.warn(
            "CuriositySiglipLoss is a legacy curiosity objective. Prefer WangIsolaUniformity for new code.",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        batch_size = g_out.size(0)
        loss = torch.zeros((), device=g_out.device, dtype=g_out.dtype)
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.calc_cross_sim and self.buffer is not None:
            elite = self._buffer_top_k_like(g_out, batch_size)
            loss = loss + sched_value * self.config.calc_cross_sim * cross_siglip(
                g_out, elite, temperature=self.config.temperature
            )

        if self.config.calc_self_sim:
            loss = loss + sched_value * self.config.calc_self_sim * self_siglip(
                g_out, temperature=self.config.temperature
            )
        return loss


@dataclass
class WangIsolaUniformityConfig(CuriosityLossBaseConfig):
    t: float = 2.0
    use_buffer: bool = False
    weight: float = 1.0


class WangIsolaUniformity(CuriosityLossBase):
    """Dedicated Wang–Isola uniformity curiosity loss."""

    def __init__(
        self,
        config: WangIsolaUniformityConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, buffer, scheduler)
        self.config = config

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        batch_size = g_out.size(0)
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.use_buffer:
            elite = self._buffer_top_k_like(g_out, batch_size)
            x = torch.cat([g_out, elite], dim=0)
        else:
            x = g_out
        return sched_value * self.config.weight * uniformity_loss(x, t=self.config.t)
