from typing import List, Sequence, Iterable
from torch import Tensor
import torch
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

    def num_levels(self) -> int:
        return len(self._names)

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

        # tensor buffer is allocated once we get the first sample
        # and where we determine shape, device and dtype from
        # NOTE: maybe we should make this configurable in case the buffer
        #       should live somewhere else then the model
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

    def get_sorted_values(self):
        return self.buffer_core.get_sorted_values()

    def _maybe_init_tensor_buffer(self, tensor: Tensor) -> None:
        """Initialize the pre-allocated tensor buffer on first tensor insertion."""
        if self.tensor_buffer is None:
            self.tensor_shape = tensor.shape
            self.device = tensor.device
            self.dtype = tensor.dtype
            # Pre-allocate buffer with zeros
            self.tensor_buffer = torch.zeros(
                (self.buffer_size,) + self.tensor_shape,
                device=self.device,
                dtype=self.dtype,
            )

    def insert(self, tensor: Tensor, value: float | Iterable[float]) -> None:
        num_levels = self.value_levels.num_levels()

        if isinstance(value, (int, float)):
            if num_levels != 1:
                raise ValueError(
                    f"Single value provided but buffer has {num_levels} levels"
                )
            value_vec = [float(value)]
        else:
            value_vec = list(map(float, value))
            if len(value_vec) != num_levels:
                raise ValueError(
                    f"Value vector length {len(value_vec)} != {num_levels}"
                )

        self._maybe_init_tensor_buffer(tensor)

        if tensor.shape != self.tensor_shape:
            raise ValueError(
                f"Tensor shape {tensor.shape} doesn't match buffer shape {self.tensor_shape}"
            )

        if (position := self.buffer_core.insert(value_vec)) is not None:
            self.tensor_buffer[position] = tensor

    def insert_many(
        self,
        tensors: List[Tensor],
        values: List[float] | List[List[float]],
    ) -> None:
        """expected layout of values:

        case 1: Iterable[float] -> obvious
        case 2: Iterable[Iterable[float]]
                e.g. two levels for batch_size 3:
                [[0.3, 0.07, 0.1], [32, 442, 132]]
                need to be inserted as:
                [0.3, 32], [0.07, 442], [0.1, 132]

        """
        if len(values) > 0 and isinstance(values[0], Iterable):
            values = list(zip(*values))

        for tensor, value_vec in zip(tensors, values):
            self.insert(tensor, value_vec)

    def get(self, idx: int | slice) -> Tensor | list[Tensor]:
        if self.tensor_buffer is None:
            raise RuntimeError("Buffer is empty – no tensors inserted yet")

        buf_len = self.buffer_core.len()
        if buf_len == 0:
            raise RuntimeError("Buffer is empty")

        sorted_indices = self.buffer_core.get_indices()
        if isinstance(idx, int):
            idx %= buf_len  # handles negative indices
            if not (0 <= idx < buf_len):
                raise IndexError(
                    f"Index {idx} out of bounds for buffer of length {buf_len}"
                )
            return self.tensor_buffer[sorted_indices[idx]]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(buf_len)
            positions = [sorted_indices[i] for i in range(start, stop, step)]
            return self.tensor_buffer[positions]

        raise TypeError("Index must be int or slice")

    def get_top_k(self, k: int) -> Tensor:
        tensors = self.get(slice(k))
        return torch.stack(tensors) if isinstance(tensors, list) else tensors

    def get_bottom_k(self, k: int) -> Tensor:
        tensors = self.get(slice(-k, None))
        return torch.stack(tensors) if isinstance(tensors, list) else tensors

    def get_top_p(self, p: float) -> Tensor:
        tensors = self.get(slice(int(p * self.buffer_core.len())))
        return torch.stack(tensors) if isinstance(tensors, list) else tensors

    def get_bottom_p(self, p: float) -> Tensor:
        tensors = self.get(slice(-int(p * self.buffer_core.len()), None))
        return torch.stack(tensors) if isinstance(tensors, list) else tensors

    def get_random_batch(self, batch_size: int) -> Tensor:
        if self.tensor_buffer is None:
            raise RuntimeError("Buffer is empty - no tensors inserted yet")
        current_len = self.buffer_core.len()
        random_positions = torch.randperm(current_len)[:batch_size]
        return self.tensor_buffer[random_positions]

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> Tensor:
        top_count = int(p * self.buffer_core.len())
        sorted_indices = self.buffer_core.get_indices()
        top_positions = sorted_indices[:top_count]
        sampled_positions = sample(top_positions, batch_size)
        return self.tensor_buffer[sampled_positions]

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> Tensor:
        sorted_indices = self.buffer_core.get_indices()
        top_positions = sorted_indices[:k]
        sampled_positions = sample(top_positions, batch_size)
        return self.tensor_buffer[sampled_positions]

    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> Tensor:
        bottom_count = int(p * self.buffer_core.len())
        sorted_indices = self.buffer_core.get_indices()
        bottom_positions = sorted_indices[-bottom_count:]
        sampled_positions = sample(bottom_positions, batch_size)
        return self.tensor_buffer[sampled_positions]

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> Tensor:
        sorted_indices = self.buffer_core.get_indices()
        bottom_positions = sorted_indices[-k:]
        assert len(bottom_positions) >= batch_size
        sampled_positions = sample(bottom_positions, batch_size)
        return self.tensor_buffer[sampled_positions]

    def get_value(self, index: int, level: int = 0) -> float:
        buf_len = self.buffer_core.len()
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

    def clear(self) -> None:
        self.buffer_core.clear()
        if self.tensor_buffer is not None:
            self.tensor_buffer.zero_()
        self.next_free_slot = 0

    def _print_value_str(self, idx: int, with_idx: bool = False) -> None:
        val = self.buffer_core.get_value(idx)
        nlevels = self.value_levels.num_levels()
        if not len(val) == nlevels:
            raise ValueError(
                "number of levels and number of values differ. This should not happen and is a bug"
            )
        vstr = "" if not with_idx else f"{idx:<12}"
        for v in val:
            vstr += f"{v:<12.5f}"
        print(vstr)

    def _print_header(self, with_idx: bool = False) -> None:
        h = "" if not with_idx else "idx".ljust(12)
        for name in self.value_levels.names():
            h += f"{name:<12}"
        print(h)

    def print_value(self, idx: int) -> None:
        self._print_header()
        self._print_value_str(idx)

    def print_values(self, idx: slice) -> None:
        start, stop, step = idx.indices(self.buffer_core.len())
        self._print_header(with_idx=True)
        for i in range(start, stop, step):
            self._print_value_str(i, with_idx=True)
