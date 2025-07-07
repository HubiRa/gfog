from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch import nn
from .buffer import BufferBase
from .utils import cross_similarity_loss
from .utils import self_similarity_loss
from .utils import cross_siglip
from .utils import self_siglip


@dataclass
class CuriosityLossBaseConfig:
    pass


class CuriosityLossBase(ABC, nn.Module):
    def __init__(
        self,
        config: CuriosityLossBaseConfig,
        buffer: BufferBase | None = None,
        **kwargs,
    ) -> None:
        self.config = config
        self.buffer = buffer

    @abstractmethod
    def forward(self, G_out: torch.Tensor) -> torch.Tensor: ...


@dataclass
class CuriosityLossConfig:
    temperature: float = 0.7

    # NOTE: a value <= 0 deactivates the loss
    calc_self_sim: float = 0.5
    calc_cross_sim: float = 0.5

    def __post_init__(self):
        if not any([v > 0 for v in [self.calc_cross_sim, self.calc_self_sim]]):
            raise ValueError(
                "At least one of calc_self_sim and calc_cross_sim must be > 0!"
            )
        self.calc_cross_sim = self.calc_cross_sim if self.calc_cross_sim > 0.0 else 0.0
        self.calc_self_sim = self.calc_self_sim if self.calc_self_sim > 0.0 else 0.0


class CuriosityLoss(CuriosityLossBase):
    """Loss is based on CLIP loss"""

    def __init__(
        self, config: CuriosityLossConfig, buffer: BufferBase | None = None, **kwargs
    ) -> None:
        self.config = config
        if self.config.calc_cross_sim and not buffer:
            raise ValueError(
                "Cross similarity loss is configured to be calculated, but buffer is not passed. Cross similarity is calculated between generator output and top K samples from the buffer"
            )
        self.buffer: BufferBase | None = buffer

    def forward(self, G_out: torch.Tensor) -> torch.Tensor:
        bs = G_out.sizse(0)
        loss = 0
        if self.config.calc_cross_sim and self.buffer:
            loss = loss + self.calc_cross_sim * cross_similarity_loss(
                G_out, self.buffer.get_top_k(bs), temperature=self.config.temperature
            )

        if self.config.calc_self_sim:
            loss = loss + self.calc_self_sim * self_similarity_loss(
                G_out, temperature=self.config.temperature
            )
        return loss


@dataclass
class CuriositySiglipLossConfig(CuriosityLossConfig):
    temperature: float = 1.0


class CuriositySiglipLoss(CuriosityLoss):
    """Loss is based on Siglip loss"""

    def forward(self, G_out: torch.Tensor) -> torch.Tensor:
        bs = G_out.sizse(0)
        loss = 0
        if self.config.calc_cross_sim and self.buffer:
            loss = loss + self.calc_cross_sim * cross_siglip(
                G_out, self.buffer.get_top_k(bs), temperature=self.config.temperature
            )

        if self.config.calc_self_sim:
            loss = loss + self.calc_self_sim * self_siglip(
                G_out, temperature=self.config.temperature
            )
        return loss
