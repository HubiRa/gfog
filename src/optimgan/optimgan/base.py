import torch
import torch.nn as nn

from loguru import logger
from abc import ABC, abstractmethod
from typing import Callable

from optimgan.buffer import Buffer


class OptimGanBase(ABC):
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
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
        curiosity: float = 1.0,
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
        self.buffer = Buffer(buffer_size, f_input_dim)
        self.init_buffer()

    @abstractmethod
    def optimize(self, x_intial: torch.Tensor) -> torch.Tensor:
        pass

    def init_sampler(self) -> torch.Tensor:
        # print(f'debbug:::self.batch_size = {self.batch_size}')
        return torch.randn(self.batch_size, self.latent_dim)

    def init_buffer(self) -> None:
        n_iter = self.buffer.buffer_size // self.batch_size
        logger.info(
            f"Filling buffer of size {self.buffer_size} with {n_iter + 1} iterations"
        )
        for i in range(n_iter + 1):
            logger.info(f"Filling buffer iteration [{i + 1}/{n_iter + 1}]")
            if self.generator is not None:
                x = self.init_sampler().to(self.device)
                with torch.no_grad():
                    x = self.generator(x)
                    # print(f'debbug:::x.shape = {x.shape}')
            else:
                x = torch.randn(self.batch_size, self.f_input_dim).to(self.device)
            values = self.f(x.to(self.f_device))
            self.buffer.insert_many(list(values), list(x.detach()))
            # print(f'# debbug: self.buffer.buffer[:15] = {self.buffer.buffer[15]}')

    def lack_of_curiosity(self, x: torch.Tensor, beta=1.0) -> torch.Tensor:
        bs = x.shape[0]
        buffer = torch.stack(self.buffer.get_top_k(bs)).to(self.device)
        x = x / x.norm(dim=-1, keepdim=True)
        buffer = buffer / buffer.norm(dim=-1, keepdim=True)
        # exploration = (beta * x @ buffer.T).softmax(dim=-1)
        exploration_loss = (beta * x @ buffer.T).mean()
        internal_curiosity = (beta * x @ x.T).softmax(dim=-1)
        # exploitation_loss = self.curiosit_loss(exploration, labels)
        # exploitation_loss = (exploration - 1./bs).abs().mean()
        internal_curiosity_loss = (internal_curiosity.diag() - 1.0).abs().mean()
        # logger.debug(f'exploration_loss = {exploration_loss}, internal_curiosity_loss = {internal_curiosity_loss}')
        return exploration_loss  # + internal_curiosity_loss

    @abstractmethod
    def step(self) -> torch.Tensor:
        pass

    def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
        pass
