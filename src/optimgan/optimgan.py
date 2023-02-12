import torch
import torch.nn as nn

from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Any, Callable, Optional

from testfunctions import TestFunctionBase
from optimgan.buffer import Buffer


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
        f: Union[Callable[[torch.Tensor], torch.Tensor], TestFunctionBase],
        f_input_dim: int,
        f_device: torch.device,
        buffer_size: int,
        generator: nn.Module = None, 
        latent_dim: int = None,
        discriminator: nn.Module = None, 
        device: torch.device = None,
        optimizerG: torch.optim.Optimizer = None,
        optimizerD: torch.optim.Optimizer = None,
        batch_size: int = None,
        curiosity: float = 1.,
        init_sampler_func: Callable[[int, int], torch.Tensor] = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.f = f
        self.f_input_dim = f_input_dim
        self.f_device = f_device
        self.buffer_size = buffer_size
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.curiosit_loss = nn.CrossEntropyLoss()
        assert not curiosity < 0
        self.curiosty = curiosity
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.buffer = Buffer(buffer_size, f_input_dim, device)
        self.init_buffer()


    @abstractmethod
    def optimize(self, x_intial: torch.Tensor) -> torch.Tensor:
        pass

    def init_sampler(self) -> torch.Tensor:
        return torch.randn(self.batch_size, self.latent_dim)

    def init_buffer(self) -> None:
        n_iter = self.buffer.buffer_size // self.batch_size
        for _ in range(n_iter + 1):
            if self.generator is not None:
                x = self.init_sampler().to(self.device)
                with torch.no_grad():
                    x = self.generator(x)
            else:
                x = torch.randn(self.batch_size, self.f_input_dim).to(self.device)
            values = self.f(x.to(self.f_device))
            self.buffer.insert_many(list(values), list(x.detach()))

    def lack_of_curiosity(self, x: torch.Tensor, beta=10.) -> torch.Tensor:
        bs = x.shape[0]
        labels = torch.arange(bs).to(x.device)
        exploration = (beta * x @ x.T).softmax(dim=-1)
        exploitation_loss = self.curiosit_loss(exploration, labels)
        return exploitation_loss

    @abstractmethod
    def step(self) -> torch.Tensor:
        pass

    def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
        pass



class RandomOpt(OptimGan):
    def step(self) -> torch.Tensor:
        x = torch.rand(self.batch_size, self.buffer.buffer_dim)
        # function evaluation
        values = self.f(x.detach().to(self.f_device))
        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))
        


class SimpleOptimGan(OptimGan):
    def step(self) -> torch.Tensor:
        # zero grad
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # train discriminator on old buffer
        good_samples = torch.stack(
            self.buffer.get_random_batch_from_top_p(
                p=0.5, batch_size=self.batch_size
            )
        ).to(self.device)
        out_buffer = self.discriminator(good_samples)
        loss_buffer = self.loss_fn(out_buffer, torch.ones_like(out_buffer))
        
        x = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            gen_samples = self.generator(x)
        out_model = self.discriminator(gen_samples.detach())
        loss_model = self.loss_fn(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.optimizerD.step()
        

        # train generator
        x = torch.randn(self.batch_size, self.latent_dim).to(self.device)
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
        loss = lossG + self.curiosty * loss_curiosity

        # backward pass
        loss.backward()

        # update generator
        self.optimizerG.step()

        # function evaluation
        values = self.f(x.detach().to(self.f_device))
        
        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))
    
    def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
        for _ in tqdm(range(n_iter)):
            self.step()
            if termination_eps is not None:
                if self.buffer.values[0] < termination_eps:
                    break
        return self.buffer.get_best_value()