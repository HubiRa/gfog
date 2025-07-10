from abc import ABC
from abc import abstractmethod
import torch
from typing import Callable


class LatentSamplerBase(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor: ...


class LatentSamplerLambda(LatentSamplerBase):
    def __init__(self, callable: Callable, **kwargs) -> None:
        self.callable = callable

    def __call__(self, **kwargs) -> torch.Tensor:
        return self.callable()
