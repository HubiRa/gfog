import torch

from .base import OptimGanBase


class RandomOpt(OptimGanBase):
    def step(self) -> torch.Tensor:
        x = torch.rand(self.batch_size, self.buffer.buffer_dim)
        # function evaluation
        values = self.f(x.detach().to(self.f_device))
        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))
