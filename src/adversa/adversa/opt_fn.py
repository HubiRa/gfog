from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class FN:
    f: Callable[[torch.Tensor], torch.Tensor]
    f_input_dim: int
    f_device: torch.device
