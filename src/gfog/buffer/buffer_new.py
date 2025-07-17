from typing import List, Sequence, Iterable
from torch import Tensor
import torch
import numpy as np
from random import sample

try:
    from buffer_core import BufferCore
except ImportError:
    print(
        "Warning: buffer_core not available. Run 'maturin develop' to build the Rust extension."
    )
    raise


class Levels:
    def __init__(self, spec: int | Sequence[str]):
        if isinstance(spec, int):
            if spec < 1:
                raise ValueError("Number of levels must be >= 1")
            self._names = [f"L{i}" for i in range(spec)]
        elif isinstance(spec, Sequence) and all(isinstance(s, str) for s in spec):
            if len(set(spec)) != len(spec):
                raise ValueError("Level names must be unique")
            self._names = list(spec)
        else:
            raise TypeError(
                "Levels must be initialized with an int or a sequence of strings"
            )

        self._name_to_index = {name: idx for idx, name in enumerate(self._names)}

    def __getitem__(self, key: int | str) -> str | int:
        if isinstance(key, int):
            return self._names[key]
        elif isinstance(key, str):
            return self._name_to_index[key]
        raise TypeError("Key must be int or str")

    def __len__(self) -> int:
        return len(self._names)

    def index(self, name: str) -> int:
        return self._name_to_index[name]

    def names(self) -> list[str]:
        return self._names.copy()

    def __repr__(self) -> str:
        return f"Levels({self._names})"


class Buffer:
    def __init__(
        self, buffer_size: int, value_levels: Levels | int = Levels(1)
    ) -> None:
        self.buffer_size = buffer_size
        self.value_levels = self._init_value_levels(value_levels)
        if BufferCore is not None:
            self.buffer_core = BufferCore(
                max_size=buffer_size, value_levels=len(self.value_levels)
            )
        else:
            raise ImportError(
                "BufferCore not available. Run 'maturin develop' to build the Rust extension."
            )

        # Pre-allocated tensor buffer - tensors stay in place, accessed via indices
        self.tensor_buffer: Tensor | None = None
        self.tensor_shape: tuple | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None

    def _init_value_levels(self, value_levels: Levels | int) -> Levels:
        if isinstance(value_levels, Levels):
            vl = value_levels
        elif isinstance(value_levels, int):
            vl = Levels(value_levels)
        else:
            raise ValueError(
                f"invalid argument type for value_levels: {type(value_levels) = }"
            )
        return vl

    def _init_tensor_buffer(self, tensor: Tensor) -> None:
        """Initialize the tensor buffer on first tensor insertion."""
        if self.tensor_buffer is None:
            self.tensor_shape = tensor.shape
            self.device = tensor.device
            self.dtype = tensor.dtype
            # Pre-allocate buffer with zeros
            self.tensor_buffer = torch.zeros(
                (self.buffer_size,) + self.tensor_shape, 
                device=self.device, 
                dtype=self.dtype
            )

    def insert(self, tensor: Tensor, value: float | Iterable[float]) -> None:
        if isinstance(value, (int, float)):
            if len(self.value_levels) != 1:
                raise ValueError(
                    f"Single value provided but buffer has {len(self.value_levels)} levels"
                )
            value_vec = [float(value)]
        else:
            value_vec = [float(v) for v in value]
            if len(value_vec) != len(self.value_levels):
                raise ValueError(
                    f"Value vector length {len(value_vec)} != {len(self.value_levels)}"
                )

        # Initialize tensor buffer if needed
        self._init_tensor_buffer(tensor)
        
        # Ensure tensor matches buffer format
        if tensor.shape != self.tensor_shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match buffer shape {self.tensor_shape}")
        
        # Insert into Rust buffer and get the index of any removed item
        removed_idx = self.buffer_core.insert(value_vec)
        
        # Find the next available slot in the tensor buffer
        current_len = self.buffer_core.len()
        if removed_idx is not None:
            # Replace the tensor at the position that was removed
            self.tensor_buffer[removed_idx] = tensor
        else:
            # Add tensor to the next position (buffer is growing)
            self.tensor_buffer[current_len - 1] = tensor

    def insert_many(
        self,
        tensors: List[Tensor],
        values: Iterable[float] | Iterable[Iterable[float]],
    ) -> None:
        values_list = list(values)
        if len(values_list) != len(tensors):
            raise ValueError(
                f"Length mismatch: {len(values_list)} values vs {len(tensors)} tensors"
            )

        # Convert values to proper format
        processed_values = []
        for i, val in enumerate(values_list):
            if isinstance(val, (int, float)):
                if len(self.value_levels) != 1:
                    raise ValueError(
                        f"Single value provided at index {i} but buffer has {len(self.value_levels)} levels"
                    )
                processed_values.append([float(val)])
            else:
                val_vec = [float(v) for v in val]
                if len(val_vec) != len(self.value_levels):
                    raise ValueError(
                        f"Value vector at index {i} has length {len(val_vec)} != {len(self.value_levels)}"
                    )
                processed_values.append(val_vec)

        # Initialize tensor buffer if needed
        if tensors:
            self._init_tensor_buffer(tensors[0])
        
        # Insert values one by one and handle tensor management
        for tensor, value_vec in zip(tensors, processed_values):
            # Ensure tensor matches buffer format
            if tensor.shape != self.tensor_shape:
                raise ValueError(f"Tensor shape {tensor.shape} doesn't match buffer shape {self.tensor_shape}")
            
            # Insert into Rust buffer and get the index of any removed item
            removed_idx = self.buffer_core.insert(value_vec)
            
            # Find the next available slot in the tensor buffer
            current_len = self.buffer_core.len()
            if removed_idx is not None:
                # Replace the tensor at the position that was removed
                self.tensor_buffer[removed_idx] = tensor
            else:
                # Add tensor to the next position (buffer is growing)
                self.tensor_buffer[current_len - 1] = tensor

    def get(self, idx: int | slice) -> Tensor | List[Tensor]:
        if isinstance(idx, int):
            if idx < 0:
                idx = self.buffer_core.len() + idx
            if idx < 0 or idx >= self.buffer_core.len():
                raise IndexError(
                    f"Index {idx} out of bounds for buffer of length {self.buffer_core.len()}"
                )

            # Get the sorted indices and find the actual tensor index
            sorted_indices = self.buffer_core.get_indices()
            actual_idx = sorted_indices[idx]
            return self.tensors[actual_idx]

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.buffer_core.len())
            sorted_indices = self.buffer_core.get_indices()
            tensors = []
            for i in range(start, stop, step):
                actual_idx = sorted_indices[i]
                tensors.append(self.tensors[actual_idx])
            return tensors
        else:
            raise TypeError("Index must be int or slice")

    def get_top_k(self, k: int) -> List[Tensor]:
        return self.get(slice(k))

    def get_bottom_k(self, k: int) -> List[Tensor]:
        return self.get(slice(-k, None))

    def get_top_p(self, p: float) -> List[Tensor]:
        return self.get(slice(int(p * self.buffer_core.len())))

    def get_bottom_p(self, p: float) -> List[Tensor]:
        return self.get(slice(-int(p * self.buffer_core.len()), None))

    def get_random_batch(self, batch_size: int) -> List[Tensor]:
        return sample(self.tensors, batch_size)

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

    def get_value(self, index: int, level: int = 0) -> float:
        if index < 0:
            index = self.buffer_core.len() + index
        if index < 0 or index >= self.buffer_core.len():
            raise IndexError(
                f"Index {index} out of bounds for buffer of length {self.buffer_core.len()}"
            )

        if level < 0:
            level = len(self.value_levels) + level
        if level < 0 or level >= len(self.value_levels):
            raise IndexError(
                f"Level index {level} out of bounds for {len(self.value_levels)} levels"
            )

        value_vec = self.buffer_core.get_value(index)
        if value_vec is None:
            raise IndexError(f"No value at index {index}")
        return value_vec[level]

    def get_mean_buffer_value(self, level: int = 0) -> float:
        if level < 0:
            level = len(self.value_levels) + level
        if level < 0 or level >= len(self.value_levels):
            raise IndexError(
                f"Level index {level} out of bounds for {len(self.value_levels)} levels"
            )
        return self.buffer_core.get_mean(level)

    def len(self) -> int:
        return self.buffer_core.len()

    def clear(self) -> None:
        self.buffer_core.clear()
        self.tensors.clear()
