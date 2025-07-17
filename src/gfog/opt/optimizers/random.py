import torch

from .base import BaseOpt


class RandomOpt(BaseOpt):
    def step(self) -> torch.Tensor:
        x = torch.rand(self.batch_size, self.buffer.buffer_dim)
        # function evaluation
        values = self.fn(x.detach().to(self.f_device))
        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))
