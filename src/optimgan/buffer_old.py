import torch
import bisect
import numpy as np

from random import sample
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Union, Iterable
from sortedcontainers import SortedSet

from loguru import logger

@dataclass
class BufferBase(ABC):
    buffer_size: int
    buffer_dim: int

    def __post_init__(self) -> None:
        pass

    @abstractmethod
    def get(self, idx: Union[int, slice]) -> torch.Tensor:
        pass

    @abstractmethod
    def insert(self, tensor: torch.Tensor, value: float) -> None:
        pass

    @abstractmethod
    def insert_many(self, tensors: List[torch.Tensor], values: List[float]) -> None:
        pass

    @abstractmethod
    def sort(self) -> None:
        pass

    @abstractmethod
    def get_top_k(self, k: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_bottom_k(self, k: int) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_top_p(self, p: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_bottom_p(self, p: float) -> torch.Tensor:
        pass

    @abstractmethod
    def get_random_batch(self, batch_size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> torch.Tensor:
        pass


class Buffer(BufferBase):
    '''
    Simple buffer that stores tensors sorted w.r.t. values
    '''
    buffer: List[torch.Tensor] = None
    values: List[float] = None

    def __post_init__(self) -> None:
        assert self.buffer_size > 0        
        assert self.buffer_dim > 0

        if self.values is None:
            self.values = [np.inf] * self.buffer_size
            self.buffer = [torch.zeros(self.buffer_dim) for _ in range(self.buffer_size)]
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


    def insert(self, value: float, tensor: torch.Tensor) -> None:
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

    def insert_many(self, values: List[float], tensors: List[torch.Tensor]) -> None:
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

    def get(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.buffer[idx]
    
    def get_top_k(self, k: int) -> List[torch.Tensor]:
        return self.get(slice(k))

    def get_bottom_k(self, k: int) -> List[torch.Tensor]:
        return self.get(slice(-k, None))
    
    def get_top_p(self, p: float) -> List[torch.Tensor]:
        return self.get(slice(int(p * self.buffer_size)))

    def get_bottom_p(self, p: float) -> List[torch.Tensor]:
        return self.get(slice(-int(p * self.buffer_size), None))

    def get_random_batch(self, batch_size: int) -> List[torch.Tensor]:
        return sample(self.buffer, batch_size)

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> List[torch.Tensor]:
        top_p = self.get_top_p(p)
        #return_len = self._check_and_maybe_set_return_len(
        #    return_len=len(top_p), batch_size=batch_size
        #)
        return sample(top_p, batch_size)

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> List[torch.Tensor]:
        top_k = self.get_top_k(k)
        return sample(top_k, batch_size)

    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> List[torch.Tensor]:
        bottom_p = self.get_bottom_p(p)
        return sample(bottom_p, batch_size)

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> List[torch.Tensor]:
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
            self.buffer = [torch.zeros(self.buffer_dim) for _ in range(self.buffer_size)]
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

    def insert(self, value: Iterable, tensor: torch.Tensor) -> None:
        ''' inserts one element into the buffer'''
        assert len(value) == self.value_levels
        idx = bisect.bisect_left(self.values, value)
        self.values.insert(idx, value)
        self.buffer.insert(idx, tensor)
        # remove worst element (TODO: change)
        self.values = self.values[:-1]
        self.buffer = self.buffer[:-1]

    def insert_many(self, values: Iterable[Iterable[float]], tensors: List[torch.Tensor]) -> None:
        '''
        naive implementation: call insert for all values
        TODO: implement less wasteful version
        '''
        # presort tensors and values
        for v, t in zip(values, tensors):
            self.insert(value=v, tensor=t)


class HirachicalValueContainer:
    '''
    This class is used to store values in a hirachical manner
    '''
    def __init__(self, buffer_size: int, levels: int=None, level_names: list[str]=None):
        buffer: List[torch.Tensor] = None
        values: List[float] = None

        self.values: dict[list[float]] = None
        self.gobal_index: torch.Tensor = torch.arange(len(buffer_size))
        self.levels: int = levels
        self.level_names: list[str] = level_names
        self._check_and_set_levels()
        self.buffer

    def _check_and_set_levels(self) -> None:
        if not any([self.levels, self.level_names]):
            raise ValueError(
                'Invalid argument combination. At least one {levels, named_lelves} required'
            )
        if all([self.levels, self.level_names]):
            if len(self.level_names) != self.levels:
                raise ValueError(
                    f'Invalid argument combination. levels: {self.levels} and len(named_levels): {self.level_names}'
                )
        if self.levels and not self.level_names:
            self.level_names = [str(i) for i in range(self.levels)]
        elif self.level_names and not self.levels:
            self.levels = len(self.level_names)

        self.values = {name:[] for name in self.level_names}


    def insert(
        self, values: list[list[float]]=None, named_values :dict[str,list]=None
    ) -> None:
        if not any([values, named_values]):
            raise ValueError(
                'Invalid argument combination: at least one of {values, named_values} required'
            )
        if all([values, named_values]):
            raise ValueError(
                'Invalid argument combination: please provide either values or named_values'
            )
        if values:
            if len(values) != len(self.values):
                raise ValueError(
                    f'Only {len(values)} value arrays provided but expeted one for each of the {self.values} levels'
                )
            for i, name in zip(range(len(values)-1,-1,-1), self.level_names[::-1]):
                self.values[name].append(values[i])
        else:
            if set(named_values.keys) != set(self.values.keys):
                raise ValueError(
                    'names of provided values do not match names of values in buffeer'
                )
            for name, value in named_values.items():
                self.values[name].append(value)

        # todo insert values
        


    def sort(self) -> None:
        ''' hirarchical sorting
        should only be called for initialization
        '''

    



    def setup():
        # TODO
        pass

@dataclass
class HirachicalBuffer(Buffer):
    '''
    BUFFER with hirachical value structure
    This is e.g. to account for constraints on the domain
    '''
    # TODO
    pass


@dataclass
class TensorBuffer(BufferBase):
    '''
    TODO. This is WIP
    Base class for buffers
    :buffer_size: size of buffer
    :buffer_dim: dimension of buffer
    :index: list where: list index corresponds to buffer index and list value 
                        corresponds to rank in sorted value list
    :values: numpy array of values retrieved from external function with buffer value as input
    '''
    buffer_index: List[int] = field(default_factory=list)
    values: torch.Tensor = field(default=None)
    buffer: torch.Tensor = field(default=None)


    def __post_init__(self) -> None:
        assert self.buffer_size > 0
        if not self.buffer_index:
            self.buffer_index = list(range(self.buffer_size))
        else:
            assert len(self.buffer_index) == self.buffer_size
        
        if self.values is None:
            self.values = torch.tensor([np.inf] * self.buffer_size)
        else:
            assert len(self.values) == self.buffer_size

        self.buffer = torch.empty(self.buffer_size, self.buffer_dim) if self.buffer is None else self.buffer
        assert self.buffer.shape == (self.buffer_size, self.buffer_dim)

        # initial sorting
        # TODO: think about this!!
        self.sort()

    def sort(self) -> None:
        '''Sort indices according to values in ascending order'''
        idx = torch.argsort(self.values)
        self.buffer_index = [self.buffer_index[i] for i in idx]
        self.values = self.values[idx]
        self.buffer = self.buffer[self.buffer_index]

    #@abstractmethod
    #def insert(self, value: float, data: torch.Tensor) -> None:
    #    pass

    def get(self, index: Union[List[int], int]) -> torch.Tensor:
        if isinstance(index, list):
            return self.buffer[index]
        else:
            return self.buffer[index]

    def insert(self, tensor: torch.tensor, values: Union[np.array, torch.Tensor]) -> None:
        """
        assumes self.buffer_index is sorted from smallest value to largest value
        assumes self.buffer is stored in the same order as self.values
        """
        # index maps from value to index in data
        # the index is sorted from smallest to largest value
        # e.g. [3,2,0,1] means that the value at index 3 is the smalles one etc
        # and the corresponding data is at index 0
        # At the beginning, the index is just the identity, since we sort the values
        # and the data is sorted in the same way
        #
        # Since the buffer size stays the same, we have to remove the largest value
        # First we check if the new value is smaller than the largest value

        # get largest value
        largest_value = self.values[-len(values):]

        # concat new values to largest values
        val = torch.cat((largest_value, values))

        # argsort concated values
        idx_new = torch.argsort(val)

        # are the values new or old? TODO: check if >= is correct
        new = idx_new[:len(values)] >= len(values)

        # we now know which of the new values should be inserted
        # if no new value should be inserted, we are done, else we need to insert some values
        if any(new):
            # there are new values to be inserted
            # get the indices of the values that should be inserted
            insert_idx = idx_new[:len(values)][new] - len(values)

            # get values
            new_values = values[insert_idx]

            # get the indices of the values that should be removed
            remove = idx_new[-len(values):] < len(values)
            remove_idx = idx_new[-len(values):][remove]
            
            # len of remove_idx should be equal to len(new_values)
            assert len(remove_idx) == len(new_values)

            # replace tensors in buffer
            self.buffer[remove_idx] = tensor[new_values]

            # replace values
            self.values[remove_idx] = new_values

            # argsort holds mapping from value to index in buffer
            self.buffer_index = torch.argsort(self.values)

            # sort values
            self.values = self.values[self.buffer_index]    

    def get_random_batch(self, batch_size: int) -> torch.Tensor:
        assert batch_size > 0 and batch_size <= len(self.buffer_index)
        return self.get(sample(self.buffer_index, batch_size))

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> torch.Tensor:
        assert p > 0 and p <= 1
        assert batch_size > 0 and batch_size <= int(len(self.buffer_index) * p)
        # get top p
        num_samples = int(len(self.buffer_index) * p)
        top_p = self.buffer_index[:num_samples]
        # get random batch from top p
        return self.get(sample(top_p, batch_size))

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.buffer_index)
        assert batch_size > 0 and batch_size <= k
        # get top k
        top_k = self.buffer_index[:k]
        # get random batch from top k
        return self.get(sample(top_k, batch_size))
        
    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> torch.Tensor:
        assert p > 0 and p <= 1
        assert batch_size > 0 and batch_size <= int(len(self.buffer_index) * p)
        # get bottom p
        num_samples = int(len(self.buffer_index) * p)
        bottom_p = self.buffer_index[-num_samples:]
        # get random batch from bottom p
        return self.get(sample(bottom_p, batch_size))

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.buffer_index)
        assert batch_size > 0 and batch_size <= k
        # get bottom k
        bottom_k = self.buffer_index[-k:]
        # get random batch from bottom k
        return self.get(sample(bottom_k, batch_size))

    def get_top_p(self, p: float) -> torch.Tensor:
        assert p > 0 and p <= 1
        num_samples = int(len(self.buffer_index) * p)
        return self.get(self.buffer_index[:num_samples])

    def get_top_k(self, k: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.buffer_index)
        return self.get(self.buffer_index[:k])

    def get_bottom_p(self, p: float) -> torch.Tensor:
        assert p > 0 and p <= 1
        num_samples = int(len(self.buffer_index) * p)
        return self.get(self.buffer_index[-num_samples:])

    def get_bottom_k(self, k: int) -> torch.Tensor:
        assert k > 0 and k <= len(self.buffer_index)
        return self.get(self.buffer_index[-k:])


if __name__ == '__main__':
    buffer = Buffer(
        buffer_size=10, 
        buffer_dim=2
    )
    print(buffer.buffer)
    buffer.insert_many(
        values=[0.4, 0.2, 0.1], 
        tensors= list(torch.tensor([[1, 2], [3, 4], [5, 6]]))
    )
    print(buffer.values)
    print(buffer.get(0))
    data = list(torch.ones(4,2)*torch.tensor([1,3,0,4]).view(-1,1))
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
    
    
