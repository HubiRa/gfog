from dataclasses import dataclass
import torch
from torch import nn
from ..buffer import Buffer
from ..utils import cross_similarity_loss
from ..utils import self_similarity_loss
from ..utils import cross_siglip
from ..utils import self_siglip
from ..utils import uniformity_loss

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


@dataclass
class CuriosityLossConfig(CuriosityLossBaseConfig):
    temperature: float = 0.7

    # NOTE: a value <= 0 deactivates the loss
    calc_self_sim: float = 0.5
    calc_cross_sim: float = 0.5

    def __post_init__(self):
        self.calc_cross_sim = self.calc_cross_sim if self.calc_cross_sim > 0.0 else 0.0
        self.calc_self_sim = self.calc_self_sim if self.calc_self_sim > 0.0 else 0.0


class CuriosityLoss(CuriosityLossBase):
    """Loss is based on CLIP loss"""

    def __init__(
        self,
        config: CuriosityLossConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, buffer, scheduler)
        self.config = config
        if self.config.calc_cross_sim and not buffer:
            raise ValueError(
                "Cross similarity loss is configured to be calculated, but buffer is not passed. Cross similarity is calculated between generator output and top K samples from the buffer"
            )
        self.buffer: Buffer | None = buffer

    def forward(self, G_out: torch.Tensor) -> torch.Tensor:
        bs = G_out.size(0)
        loss = 0
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.calc_cross_sim and self.buffer:
            loss = (
                loss
                + sched_value
                * self.config.calc_cross_sim
                * cross_similarity_loss(
                    G_out,
                    self.buffer.get_top_k(bs),
                    temperature=self.config.temperature,
                )
            )

        if self.config.calc_self_sim:
            loss = (
                loss
                + sched_value
                * self.config.calc_self_sim
                * self_similarity_loss(G_out, temperature=self.config.temperature)
            )
        return loss


@dataclass
class CuriositySiglipLossConfig(CuriosityLossConfig):
    temperature: float = 1.0


class CuriositySiglipLoss(CuriosityLoss):
    """Loss is based on Siglip loss"""

    def forward(self, G_out: torch.Tensor) -> torch.Tensor:
        bs = G_out.size(0)
        loss = 0
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.calc_cross_sim and self.buffer:
            loss = loss + sched_value * self.config.calc_cross_sim * cross_siglip(
                G_out, self.buffer.get_top_k(bs), temperature=self.config.temperature
            )

        if self.config.calc_self_sim:
            loss = loss + sched_value * self.config.calc_self_sim * self_siglip(
                G_out, temperature=self.config.temperature
            )
        return loss


@dataclass
class WangIsolaUniformityConfig(CuriosityLossBaseConfig):
    t: float = 2.0
    use_buffer: bool = False
    weight: float = 1.0


class WangIsolaUniformity(CuriosityLossBase):
    """Dedicated Wang–Isola uniformity curiosity loss.

    Computes log E[exp(-t * ||xi - xj||^2)] on L2‑normalized embeddings. Optionally
    augments the batch with top‑K items from the buffer.
    """

    def __init__(
        self,
        config: WangIsolaUniformityConfig,
        buffer: Buffer | None = None,
        scheduler: Scheduler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, buffer, scheduler)
        self.config = config

    def forward(self, G_out: torch.Tensor) -> torch.Tensor:
        bs = G_out.size(0)
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        if self.config.use_buffer:
            if not self.buffer:
                raise ValueError("use_buffer=True but no buffer provided")
            x = torch.cat([G_out, self.buffer.get_top_k(bs)], dim=0)
        else:
            x = G_out
        return sched_value * self.config.weight * uniformity_loss(
            x, t=self.config.t
        )
