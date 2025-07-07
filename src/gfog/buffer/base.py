import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Iterable

from random import sample
from torch import Tensor


@dataclass
class BufferBase(ABC):
    buffer_size: int
    buffer_dim: int
    buffer: List[Tensor] | None = None
    values: List[float] | None = None

    def __post_init__(self) -> None:
        assert self.buffer_size > 0
        assert self.buffer_dim > 0

        if self.values is None:
            self.values = [np.inf] * self.buffer_size
            self.buffer = [np.nan for _ in range(self.buffer_size)]
        else:
            assert len(self.values) == self.buffer_size

        self._sort()

    def _sort(self) -> None:
        """
        should only be called for initialization
        """
        # self.values, self.buffer = zip(*sorted(zip(self.values, self.buffer)))
        idx_sorted = np.argsort(self.values)
        self.values = [self.values[i] for i in idx_sorted]
        self.buffer = [self.buffer[i] for i in idx_sorted]

    @abstractmethod
    def insert(self, tensor: Tensor, value: Union[float, Iterable[float]]) -> None: ...

    @abstractmethod
    def insert_many(
        self,
        tensors: List[Tensor],
        values: Union[Iterable[float], Iterable[Iterable[float]]],
    ) -> None: ...

    def get(self, idx: Union[int, slice]) -> Tensor:
        return self.buffer[idx]

    def get_top_k(self, k: int) -> List[Tensor]:
        return self.get(slice(k))

    def get_bottom_k(self, k: int) -> List[Tensor]:
        return self.get(slice(-k, None))

    def get_top_p(self, p: float) -> List[Tensor]:
        return self.get(slice(int(p * self.buffer_size)))

    def get_bottom_p(self, p: float) -> List[Tensor]:
        return self.get(slice(-int(p * self.buffer_size), None))

    def get_random_batch(self, batch_size: int) -> List[Tensor]:
        return sample(self.buffer, batch_size)

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> List[Tensor]:
        top_p = self.get_top_p(p)
        return sample(top_p, batch_size)

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> List[Tensor]:
        top_k = self.get_top_k(k)
        return sample(top_k, batch_size)

    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> List[Tensor]:
        bottom_p = self.get_bottom_p(p)
        return sample(bottom_p, batch_size)

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> List[Tensor]:
        bottom_k = self.get_bottom_k(k)
        assert len(bottom_k) >= batch_size
        return sample(bottom_k, batch_size)

    def get_mean_buffer_value(self) -> float:
        return np.mean(self.values)
