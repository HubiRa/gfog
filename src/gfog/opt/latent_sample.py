from abc import ABC
from abc import abstractmethod
import torch


class LatentSamplerBase(ABC):
    def __init__(self, latent_dim: torch.Size, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self) -> torch.Tensor: ...
