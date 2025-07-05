import bisect
import numpy as np

from random import sample
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Union, Iterable, TypeVar, Generic
from sortedcontainers import SortedSet

from loguru import logger

Tensor = TypeVar('Tensor')  # Can be any type

@dataclass
class BufferBase(ABC, Generic[Tensor]):
    buffer_size: int
    buffer_dim: int

    def __post_init__(self) -> None:
        pass

    @abstractmethod
    def get(self, idx: Union[int, slice]) -> Tensor:
        pass

    @abstractmethod
    def insert(self, tensor: Tensor, value: Union[float, Iterable[float]]) -> None:
        pass

    @abstractmethod
    def insert_many(
        self, 
        tensors: List[Tensor], 
        values: Union[Iterable[float], Iterable[Iterable[float]]]
    ) -> None:
        pass

    @abstractmethod
    def sort(self) -> None:
        pass

    @abstractmethod
    def get_top_k(self, k: int) -> Tensor:
        pass

    @abstractmethod
    def get_bottom_k(self, k: int) -> Tensor:
        pass
    
    @abstractmethod
    def get_top_p(self, p: float) -> Tensor:
        pass

    @abstractmethod
    def get_bottom_p(self, p: float) -> Tensor:
        pass

    @abstractmethod
    def get_random_batch(self, batch_size: int) -> Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> Tensor:
        pass


class Buffer(BufferBase):
    '''
    Simple buffer that stores tensors sorted w.r.t. values
    '''
    buffer: List[Tensor] = None
    values: List[float] = None

    def __post_init__(self) -> None:
        assert self.buffer_size > 0        
        assert self.buffer_dim > 0

        if self.values is None:
            self.values = [np.inf] * self.buffer_size
            self.buffer = [np.nan for _ in range(self.buffer_size)]
        else:
            assert len(self.values) == self.buffer_size

        self.sort()

    def sort(self) -> None:
        '''
        should only be called for initialization
        '''
        # self.values, self.buffer = zip(*sorted(zip(self.values, self.buffer)))
        idx_sorted = np.argsort(self.values)
        self.values = [self.values[i] for i in idx_sorted]
        self.buffer = [self.buffer[i] for i in idx_sorted]


    def insert(self, value: float, tensor: Tensor) -> None:
        ''' inserts one element into the buffer'''
        if len(self.buffer) == self.buffer_size:
            # if the buffer is full, remove the largest value
            if value < self.values[-1]:
                self.buffer.pop(-1)
                self.values.pop(-1)
            else:
                return
        idx = bisect.bisect_left(self.values, value)
        self.values.insert(idx, value)
        self.buffer.insert(idx, tensor)

    def insert_many(self, values: List[float], tensors: Iterable[Tensor]) -> None:
        '''
        presort: if True, tensors and values are sorted before insertion to avoid unnecessary inserts
        TODO: check if presort is necessary in terms of efficiency
        '''
        # presort tensors and values
        idx_sorted = np.argsort(values)
        values = [values[i] for i in idx_sorted]
        tensors = [tensors[i] for i in idx_sorted]


        for tensor, value in zip(tensors, values):
            if value > self.values[-1]:
                # we are done if one value is greater than the
                # greatest value in the buffer
                break
            self.insert(value=value, tensor=tensor)

    def _check_and_maybe_set_return_len(
            self, return_len: int, batch_size: int
    ) -> int:
        if return_len > batch_size:
            logger.warning(f'return_len {return_len} > batch_size {batch_size}'
                           f'setting return_len to {batch_size}')
            return_len = batch_size
        return return_len

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


class HirarchicalySortedBuffer(Buffer):
    '''
    sortes buffer according to a hirarchy of values
    '''   
    value_levels: int

    def __post_init__(self) -> None:
        assert self.buffer_size > 0        
        assert self.buffer_dim > 0
        assert self.value_levels > 0

        if self.values is None:
            self.values = [(np.inf)*self.value_levels] * self.buffer_size
            self.buffer = [np.nan for _ in range(self.buffer_size)]
        else:
            assert len(self.values) == self.buffer_size
            assert all(len(v) == self.value_levels for v in self.values)

        self.sort()

    def sort(self) -> None:
        '''
        should only be called for initialization
        '''
        idx_sorted = sorted(range(len(self.values)), key=lambda i: self.values[i])
        self.values = [self.values[i] for i in idx_sorted]
        self.buffer = [self.buffer[i] for i in idx_sorted]

    def insert(self, value: Iterable, tensor: Tensor) -> None:
        ''' inserts one element into the buffer'''
        assert len(value) == self.value_levels
        idx = bisect.bisect_left(self.values, value)
        self.values.insert(idx, value)
        self.buffer.insert(idx, tensor)
        # remove worst element (TODO: change)
        self.values = self.values[:-1]
        self.buffer = self.buffer[:-1]

    def insert_many(self, values: Iterable[Iterable[float]], tensors: List[Tensor]) -> None:
        '''
        naive implementation: call insert for all values
        TODO: implement less wasteful version
        '''
        # presort tensors and values
        for v, t in zip(values, tensors):
            self.insert(value=v, tensor=t)

    def get_mean_buffer_value(self, level: int=-1) -> float:
        level_values = [v[level] for v in self.values]
        return np.mean(level_values)


if __name__ == '__main__':
    buffer = Buffer(
        buffer_size=10, 
        buffer_dim=2
    )
    print(buffer.buffer)
    buffer.insert_many(
        values=[0.4, 0.2, 0.1], 
        tensors= list(Tensor([[1, 2], [3, 4], [5, 6]]))
    )
    print(buffer.values)
    print(buffer.get(0))
    data = list(torch.ones(4,2)*Tensor([1,3,0,4]).view(-1,1))
    buffer.insert_many(
        values=[1, 3, 0, 4], 
        tensors=data
    )
    print(buffer.get_top_k(k=2))
    print(buffer.get_bottom_k(k=2))
    print(buffer.get_random_batch(batch_size=2))
    print(buffer.get_random_batch_from_top_p(p=0.5, batch_size=2))
    print(buffer.get_random_batch_from_top_k(k=5, batch_size=2))
    print(buffer.get_random_batch_from_bottom_p(p=0.5, batch_size=2))
    print(buffer.get_random_batch_from_bottom_k(k=5, batch_size=2))
    
    
