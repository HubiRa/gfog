from collections.abc import Iterable
from random import sample
from typing import List

import torch
from torch import Tensor

from .levels import Levels

try:
    from buffer_core import BufferCore
except ImportError:
    print(
        "Warning: buffer_core not available. Run 'maturin develop' to build the Rust extension."
    )
    raise


class Buffer:
    def __init__(
        self, buffer_size: int, value_levels: Levels | int = Levels(1)
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be > 0, got {buffer_size}")
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

        self.tensor_buffer: Tensor | None = None
        self.tensor_shape: tuple[int, ...] | None = None
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None

    def _init_value_levels(self, value_levels: Levels | int) -> Levels:
        if isinstance(value_levels, Levels):
            return value_levels
        if isinstance(value_levels, int):
            return Levels(value_levels)
        raise ValueError(
            f"invalid argument type for value_levels: {type(value_levels) = }"
        )

    def get_sorted_values(self) -> list[list[float]]:
        return self.buffer_core.get_sorted_values()

    def _maybe_init_tensor_buffer(self, tensor: Tensor) -> None:
        if self.tensor_buffer is None:
            self.tensor_shape = tuple(tensor.shape)
            self.device = tensor.device
            self.dtype = tensor.dtype
            self.tensor_buffer = torch.zeros(
                (self.buffer_size,) + self.tensor_shape,
                device=self.device,
                dtype=self.dtype,
            )

    def insert(self, tensor: Tensor, value: float | Iterable[float]) -> None:
        value_vec = self.value_levels.expand_input(value)
        self._maybe_init_tensor_buffer(tensor)

        if tensor.shape != self.tensor_shape:
            raise ValueError(
                f"Tensor shape {tensor.shape} doesn't match buffer shape {self.tensor_shape}"
            )

        if (position := self.buffer_core.insert(value_vec)) is not None:
            assert self.tensor_buffer is not None
            self.tensor_buffer[position] = tensor

    def _normalize_many_values(
        self,
        tensors: List[Tensor],
        values: List[float] | List[List[float]],
    ) -> list[float | list[float] | tuple[float, ...]]:
        if (
            len(values) != len(tensors)
            and len(values) != self.value_levels.num_levels()
        ):
            raise ValueError(
                "values must either be row-major with one entry per tensor or "
                "column-major with one entry per value level"
            )
        if len(values) == 0:
            return []

        first_value = values[0]
        is_scalar_like = not isinstance(first_value, Iterable) or isinstance(
            first_value, (str, bytes)
        )
        if isinstance(first_value, torch.Tensor) and first_value.dim() == 0:
            is_scalar_like = True

        if is_scalar_like:
            if len(values) != len(tensors):
                raise ValueError(
                    "Flat values must contain exactly one scalar value per tensor"
                )
            return list(values)

        nested_values = [list(v) for v in values]

        if len(nested_values) == len(tensors):
            return nested_values

        if len(nested_values) == self.value_levels.num_levels() and all(
            len(v) == len(tensors) for v in nested_values
        ):
            return [list(v) for v in zip(*nested_values)]

        raise ValueError(
            "Could not infer values layout. Use row-major [[sample levels...], ...] or "
            "column-major [[level batch...], ...]."
        )

    def insert_many(
        self,
        tensors: List[Tensor],
        values: List[float] | List[List[float]],
    ) -> None:
        """Insert many tensor/value pairs.

        Accepted value layouts:
        - flat scalar list for single-level buffers: [v1, v2, ...]
        - row-major: [[sample1_levels...], [sample2_levels...], ...]
        - column-major: [[level1_batch...], [level2_batch...], ...]
        """
        normalized_values = self._normalize_many_values(tensors, values)
        if len(normalized_values) != len(tensors):
            raise ValueError(
                f"Number of values ({len(normalized_values)}) does not match number of tensors ({len(tensors)})"
            )
        for tensor, value_vec in zip(tensors, normalized_values):
            self.insert(tensor, value_vec)

    def get(self, idx: int | slice) -> Tensor:
        if self.tensor_buffer is None:
            raise RuntimeError("Buffer is empty – no tensors inserted yet")

        buf_len = self.buffer_core.len()
        if buf_len == 0:
            raise RuntimeError("Buffer is empty")

        sorted_indices = self.buffer_core.get_indices()
        if isinstance(idx, int):
            idx %= buf_len
            return self.tensor_buffer[sorted_indices[idx]]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(buf_len)
            positions = [sorted_indices[i] for i in range(start, stop, step)]
            return self.tensor_buffer[positions]

        raise TypeError("Index must be int or slice")

    def __getitem__(self, idx: int | slice) -> Tensor:
        return self.get(idx)

    def get_top_k(self, k: int) -> Tensor:
        k = min(max(k, 0), len(self))
        return self.get(slice(0, k, 1))

    def get_bottom_k(self, k: int) -> Tensor:
        k = min(max(k, 0), len(self))
        if self.tensor_buffer is None:
            raise RuntimeError("Buffer is empty – no tensors inserted yet")
        if k == 0:
            return self.tensor_buffer[:0]
        return self.get(slice(len(self) - k, len(self), 1))

    def get_top_p(self, p: float) -> Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_top_k(int(p * len(self)))

    def get_bottom_p(self, p: float) -> Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_bottom_k(int(p * len(self)))

    def get_random_batch(self, batch_size: int) -> Tensor:
        if self.tensor_buffer is None:
            raise RuntimeError("Buffer is empty - no tensors inserted yet")
        current_len = len(self)
        if batch_size > current_len:
            raise ValueError(
                f"batch_size {batch_size} exceeds current buffer length {current_len}"
            )
        random_positions = torch.randperm(
            current_len, device=self.tensor_buffer.device
        )[:batch_size]
        sorted_indices = torch.tensor(
            self.buffer_core.get_indices(), device=self.tensor_buffer.device
        )
        return self.tensor_buffer[sorted_indices[random_positions]]

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        top_count = int(p * len(self))
        return self.get_random_batch_from_top_k(top_count, batch_size)

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> Tensor:
        sorted_indices = self.buffer_core.get_indices()
        top_positions = sorted_indices[:k]
        if len(top_positions) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from top_k={len(top_positions)}"
            )
        sampled_positions = sample(top_positions, batch_size)
        assert self.tensor_buffer is not None
        return self.tensor_buffer[sampled_positions]

    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        bottom_count = int(p * len(self))
        return self.get_random_batch_from_bottom_k(bottom_count, batch_size)

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> Tensor:
        sorted_indices = self.buffer_core.get_indices()
        bottom_positions = sorted_indices[-k:]
        if len(bottom_positions) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from bottom_k={len(bottom_positions)}"
            )
        sampled_positions = sample(bottom_positions, batch_size)
        assert self.tensor_buffer is not None
        return self.tensor_buffer[sampled_positions]

    def get_value(self, index: int, level: int = 0) -> float:
        buf_len = len(self)
        lvl_len = self.value_levels.num_levels()

        if buf_len == 0:
            raise IndexError("Buffer is empty")
        if lvl_len == 0:
            raise IndexError("No value levels available")

        index %= buf_len
        level %= lvl_len

        value_vec = self.buffer_core.get_value(index)
        if value_vec is None:
            raise IndexError(f"No value at index {index}")

        return value_vec[level]

    def get_mean_buffer_value(self, level: int = 0) -> float:
        if level < 0:
            level = self.value_levels.num_levels() + level
        if level < 0 or level >= self.value_levels.num_levels():
            raise IndexError(
                f"Level index {level} out of bounds for {self.value_levels.num_levels()} levels"
            )
        return self.buffer_core.get_mean(level)

    def len(self) -> int:
        return self.buffer_core.len()

    def __len__(self) -> int:
        return self.len()

    def clear(self) -> None:
        self.buffer_core.clear()
        self.tensor_buffer = None
        self.tensor_shape = None
        self.device = None
        self.dtype = None

    def _print_value_str(self, idx: int, with_idx: bool = False) -> None:
        val = self.buffer_core.get_value(idx)
        nlevels = self.value_levels.num_levels()
        if val is None or len(val) != nlevels:
            raise ValueError(
                "number of levels and number of values differ. This should not happen and is a bug"
            )
        value_string = "" if not with_idx else f"{idx:<12}"
        for value in val:
            value_string += f"{value:<12.5f}"
        print(value_string)

    def _print_header(self, with_idx: bool = False) -> None:
        header = "" if not with_idx else "idx".ljust(12)
        for name in self.value_levels.names():
            header += f"{name:<12}"
        print(header)

    def print_value(self, idx: int) -> None:
        self._print_header()
        self._print_value_str(idx)

    def print_values(self, idx: slice) -> None:
        start, stop, step = idx.indices(len(self))
        self._print_header(with_idx=True)
        for i in range(start, stop, step):
            self._print_value_str(i, with_idx=True)
