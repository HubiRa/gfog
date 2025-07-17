from .simple import SimpleBuffer
import bisect
import numpy as np

from torch import Tensor
from typing import List, Iterable

# from dataclasses import dataclass
import attr
from loguru import logger
from tqdm import tqdm


@attr.define(kw_only=True)
class HirarchicalySortedBuffer(SimpleBuffer):
    """
    sortes buffer according to a hirarchy of values
    """

    # TODO: revisit whole implementation
    value_levels: int

    def __post_init__(self) -> None:
        assert self.buffer_size > 0
        assert self.buffer_dim > 0
        assert self.value_levels > 0

        if self.values is None:
            self.values = [(np.inf) * self.value_levels] * self.buffer_size
            self.buffer = [np.nan for _ in range(self.buffer_size)]
        else:
            assert len(self.values) == self.buffer_size
            assert all(len(v) == self.value_levels for v in self.values)

        self._sort()

    def _sort(self) -> None:
        """
        should only be called for initialization
        """
        idx_sorted = sorted(range(len(self.values)), key=lambda i: self.values[i])
        self.values = [self.values[i] for i in idx_sorted]
        self.buffer = [self.buffer[i] for i in idx_sorted]

    def insert(self, tensor: Tensor, value: Iterable[float]) -> None:
        """inserts one element into the buffer"""
        self.buffer: List[Tensor]
        self.values: List[float]
        assert len(value) == self.value_levels
        idx = bisect.bisect_left(self.values, value)
        self.values.insert(idx, value)
        self.buffer.insert(idx, tensor)
        # remove worst element (TODO: change)
        self.values = self.values[:-1]
        self.buffer = self.buffer[:-1]

    def insert_many(
        self,
        tensors: List[Tensor],
        values: Iterable[Iterable[float]],
    ) -> None:
        """
        naive implementation: call insert for all values
        TODO: implement less wasteful version
        """
        if not values:
            raise ValueError("values array is empty")
        # presort tensors and values
        if not isinstance(values[0], Iterable):
            raise ValueError(
                "hirarchical buffer expects multiple value levels. If the function only returns one value, consider using SimpleBuffer"
            )

        for v, t in zip(values, tensors):
            self.insert(value=v, tensor=t)

    def get_mean_buffer_value(self, level: int = -1) -> float:
        level_values = [v[level] for v in self.values]
        return np.mean(level_values)
