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
        optimizerG: torch.optim.Optimizer = torch.optim.Adam,
        optimizerD: torch.optim.Optimizer = torch.optim.Adam,
        batch_size: int = 32,
        latent_dim: int = 100
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.f = f
        self.f_device = f_device
        self.buffer = buffer
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.curiosit_loss = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()

    @abstractmethod
    def optimize(self, x_intial: torch.Tensor) -> torch.Tensor:
        pass

    def init_buffer(self, x: torch.Tensor) -> None:
        n_iter = self.buffer.buffer_size // self.batch_size
        for _ in range(n_iter + 1):
            x = torch.randn(self.batch_size, self.latent_dim).to(self.device)
            x = self.generator(x)
            values = self.f(x)
            self.buffer.insert_many(list(values), list(x))

    def lack_of_curiosity(self, x: torch.Tensor, beta=10.) -> torch.Tensor:
        bs = x.shape[0]
        labels = torch.arange(bs).to(x.device)
        exploration = (beta * x @ x.T).softmax(dim=-1)
        exploitation_loss = self.curiosit_loss(exploration, labels)
        return exploitation_loss




class SimpleOptimGan(OptimGan):
    def step(self) -> torch.Tensor:
        
        # train discriminator on old buffer
        good_samples = self.buffer.get_random_batch_from_top_p(
            p=0.5, batch_size=self.batch_size
            ).to(self.device)
        out_buffer = self.discriminator(good_samples)
        loss_buffer = self.loss_fn(out_buffer, torch.ones_like(out_buffer))
        
        x = torch.randn(self.batch_size, self.buffer.buffer_dim).to(self.device)
        x = x.to(self.device)
        out_model = self.discriminator(x.detach())
        loss_model = self.loss_fn(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.optimizerD.step()
        
        # train generator
        x = torch.randn(self.batch_size, self.buffer.buffer_dim).to(self.device)
        x = x.to(self.device)

        # forward pass
        x = self.generator(x)

        # panalize lack of curiousity
        loss_curiosity = self.lack_of_curiosity(x)

        # get disciminator output
        outG = self.discriminator(x)

        # calculate loss
        lossG = self.loss_fn(outG, torch.ones_like(outG))

        # total loss
        loss = lossG + loss_curiosity

        # backward pass
        loss.backward()

        # update generator
        self.optimizerG.step()

        # function evaluation
        values = self.f(x.detach())
        
        # add values to buffer
        self.buffer.insert_many(list(values), list(x))
        

    
