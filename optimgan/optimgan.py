import torch
import torch.nn as nn

from typing import List, Tuple, Union, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod


class OptimGan(ABC):
    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, device: torch.device
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    @abstractmethod
    @classmethod

