import torch

from loguru import logger
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..components import OptComponents


class BaseOpt(ABC):
    def __init__(self, components: OptComponents) -> None:
        self.components = components
        # convenience:
        self.fn = self.components.fn
        self.gan = self.components.gan
        self.buffer = self.components.buffer
        self.init_buffer()

    def init_buffer(self) -> None:
        n_iter = self.buffer.B.buffer_size // self.components.batch_size
        logger.info(
            f"Filling buffer of size {self.buffer.B.buffer_size} with {n_iter + 1} iterations"
        )
        for _ in tqdm(range(n_iter + 1)):
            if self.gan.G is not None:
                x = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
                with torch.no_grad():
                    x = self.gan.G(x)
            else:
                x = torch.randn(self.components.batch_size, self.fn.input_dim)

            values = self.fn.f(x.to(self.fn.device, self.fn.dtype))
            # breakpoint()
            self.buffer.B.insert_many(values=list(values), tensors=list(x.detach()))

        @abstractmethod
        def step(self) -> None:
            pass

        def optimize(
            self, n_iter: int, termination_eps: float | None = None
        ) -> torch.Tensor:
            pass
