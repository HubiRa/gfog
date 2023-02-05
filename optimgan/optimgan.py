import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Any, Callable, Optional

from testfunctions import TestFunctionBase
from buffer import Buffer


class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int], 
        activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = self.activation(x)
        return x


class OptimGan(ABC):
    def __init__(
        self, 
        generator: nn.Module, 
        discriminator: nn.Module, 
        device: torch.device,
        f: Union[Callable[[torch.Tensor], torch.Tensor], TestFunctionBase],
        f_device: torch.device,
        buffer: Buffer,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.f = f
        self.f_device = f_device
        self.buffer = buffer
        self.optimizer = optimizer

    @abstractmethod
    def optimize(self, x_intial: torch.Tensor) -> torch.Tensor:
        pass



class SimpleOptimGan(OptimGan):
    def optimize(self, x_intial: torch.Tensor) -> torch.Tensor:
        x = x_intial.to(self.device)
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=0.1)
        for _ in range(100):
            optimizer.zero_grad()
            x = self.generator(x)
            x = self.f(x)
            x.backward()
            optimizer.step()
        return x.detach().to(self.f_device)


    
