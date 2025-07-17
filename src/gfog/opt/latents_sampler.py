from abc import ABC
from abc import abstractmethod
import torch
from typing import Callable


class LatentSamplerBase(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self) -> torch.Tensor: ...


class LatentSamplerLambda(LatentSamplerBase):
    def __init__(self, f: Callable, **kwargs) -> None:
        self.f = f
        self.kwargs = kwargs

    def __call__(self) -> torch.Tensor:
        return self.f(**self.kwargs)
