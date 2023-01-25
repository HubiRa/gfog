import torch
import numpy as np

from random import sample
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class BufferBase(ABC):
    buffer_size: int 
    index: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.buffer_size > 0
        if not self.index:
            self.index = list(range(self.buffer_size))
        else:
            assert len(self.index) == self.buffer_size
        
        if not self.values:
            self.values = [np.inf] * self.buffer_size
        else:
            assert len(self.values) == self.buffer_size
            self.sort()

    def sort(self) -> None:
        '''Sort indices according to values in ascending order'''
        idx = np.argsort(self.values)
        self.index = [self.index[i] for i in idx]
        self.values = [self.values[i] for i in idx]

    @abstractmethod
    def update(self, index: int, value: float, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get(self, index: Union[List[int], int]) -> torch.Tensor:
        pass

    @abstractmethod
    def insert(self, value: float, data: torch.Tensor) -> None:
        pass

    def get_random_batch(self, batch_size: int) -> torch.Tensor:
        assert batch_size > 0 and batch_size <= len(self.index)
        return self.get(sample(self.index, batch_size))

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> torch.Tensor:
        assert p > 0 and p <= 1
        assert batch_size > 0 and batch_size <= int(len(self.index) * p)
        # get top p
        num_samples = int(len(self.index) * p)
        top_p = self.index[:num_samples]
        # get random batch from top p
        return self.get(sample(top_p, batch_size))

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.index)
        assert batch_size > 0 and batch_size <= k
        # get top k
        top_k = self.index[:k]
        # get random batch from top k
        return self.get(sample(top_k, batch_size))
        
    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> torch.Tensor:
        assert p > 0 and p <= 1
        assert batch_size > 0 and batch_size <= int(len(self.index) * p)
        # get bottom p
        num_samples = int(len(self.index) * p)
        bottom_p = self.index[-num_samples:]
        # get random batch from bottom p
        return self.get(sample(bottom_p, batch_size))

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.index)
        assert batch_size > 0 and batch_size <= k
        # get bottom k
        bottom_k = self.index[-k:]
        # get random batch from bottom k
        return self.get(sample(bottom_k, batch_size))

    def get_top_p(self, p: float) -> torch.Tensor:
        assert p > 0 and p <= 1
        num_samples = int(len(self.index) * p)
        return self.get(self.index[:num_samples])

    def get_top_k(self, k: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.index)
        return self.get(self.index[:k])

    def get_bottom_p(self, p: float) -> torch.Tensor:
        assert p > 0 and p <= 1
        num_samples = int(len(self.index) * p)
        return self.get(self.index[-num_samples:])

    def get_bottom_k(self, k: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.index)
        return self.get(self.index[-k:])

@dataclass
class BufferSimple(BufferBase):
    data: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.data = [torch.empty(0)] * self.buffer_size

    def get(self, index: Union[List[int], int]) -> torch.Tensor:
        if isinstance(index, list):
            return torch.stack([self.data[i] for i in index])
        else:
            return self.data[index]

    def insert(self, value: float, data: torch.Tensor) -> None:
        self.values.append(value)
        idx = np.argsort(self.values)[-1]
        idx
        
        


