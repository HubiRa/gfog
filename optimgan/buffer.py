import torch
import bisect
import numpy as np

from random import sample
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class BufferBase(ABC):
    @abstractmethod
    def get(self, idx: Union[int, slice]) -> torch.Tensor:
        pass

    @abstractmethod
    def insert(self, tensor: torch.Tensor, value: float) -> None:
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
    buffer_size: int
    buffer_dim: int
    buffer: List[torch.Tensor] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.buffer_size > 0        
        assert self.buffer_dim > 0

        if self.values is None:
            self.values = torch.tensor([np.inf] * self.buffer_size)
            self.buffer = [torch.zeros(self.buffer_dim) for _ in range(self.buffer_size)]
        else:
            assert len(self.values) == self.buffer_size

        self.sort()

    def sort(self) -> None:
        '''
        should only be called when for initialization
        '''
        self.values, self.buffer = zip(*sorted(zip(self.values, self.buffer)))


    def insert(self, tensor: torch.Tensor, value: float) -> None:
        if len(self.buffer) == self.size:
            # if the buffer is full, remove the smallest value
            if self.values[-1] < value:
                self.buffer.pop(-1)
                self.values.pop(-1)
            else:
                return
        idx = bisect.bisect_left(self.values, value)
        self.values.insert(idx, value)
        self.buffer.insert(idx, tensor)

    def insert_many(self, tensors: List[torch.Tensor], values: List[float]) -> None:
        for tensor, value in zip(tensors, values):
            self.insert(tensor, value)

    def get(self, idx: Union[int, slice]) -> torch.Tensor:
        return self.buffer[idx]
    
    def get_top_k(self, k: int) -> torch.Tensor:
        return torch.Tensor(self.get(slice(k)))

    def get_bottom_k(self, k: int) -> torch.Tensor:
        return torch.Tensor(self.get(slice(-k, None)))

@dataclass
class TensorBuffer(BufferBase):
    '''
    Base class for buffers
    :buffer_size: size of buffer
    :buffer_dim: dimension of buffer
    :index: list where: list index corresponds to buffer index and list value 
                        corresponds to rank in sorted value list
    :values: numpy array of values retrieved from external function with buffer value as input
    '''
    buffer_size: int
    buffer_dim: int 
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
    sbuff = TensorBuffer(
        buffer_size=10, 
        buffer_dim=2
    )
    #sbuff.insert([0.4, 0.2, 0.1], torch.tensor([[1, 2], [3, 4], [5, 6]]))
    print(sbuff.data_index)
    print(sbuff.values)
    print(sbuff.get([0,1,2]))
    data = torch.ones(4,2)*torch.tensor([1,3,0,4]).view(-1,1)
    sbuff.insert(torch.Tensor([1, 3, 0, 4]), data)
    print(sbuff.get_top_k(k=2))
    
