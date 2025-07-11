import torch

from loguru import logger
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..components import OptComponents


class BaseOpt(ABC):
    def __init__(self, components: OptComponents) -> None:
        self.components = components
        # convenience:
        self.f = self.components.fn
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
                x = torch.randn(self.components.batch_size, self.f.input_dim)

            values = self.f.f(x.to(self.f.device, self.f.dtype))
            self.buffer.B.insert_many(list(values), list(x.detach()))

        @abstractmethod
        def step(self) -> torch.Tensor:
            pass

        def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
            pass


# class BaseOpt(ABC):
#     def __init__(
#         self,
#         f: Callable[[torch.Tensor], torch.Tensor],
#         f_input_dim: int,
#         f_device: torch.device,
#         buffer_size: int,
#         generator: nn.Module = None,
#         latent_dim: int = None,
#         discriminator: nn.Module = None,
#         device: torch.device = None,
#         optimizerG: torch.optim.Optimizer = None,
#         optimizerD: torch.optim.Optimizer = None,
#         batch_size: int = None,
#         curiosity: float = 1.0,
#         init_sampler_func: Callable[[int, int], torch.Tensor] = None,
#     ) -> None:
#         self.generator = generator
#         self.discriminator = discriminator
#         self.device = device
#         self.f = f
#         self.f_input_dim = f_input_dim
#         self.f_device = f_device
#         self.buffer_size = buffer_size
#         self.optimizerG = optimizerG
#         self.optimizerD = optimizerD
#         self.batch_size = batch_size
#         self.latent_dim = latent_dim
#         self.curiosit_loss = nn.CrossEntropyLoss()
#         assert not curiosity < 0
#         self.curiosty = curiosity
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.buffer = Buffer(buffer_size, f_input_dim)
#         self.init_buffer()
#
#     def init_sampler(self) -> torch.Tensor:
#         # print(f'debbug:::self.batch_size = {self.batch_size}')
#         return torch.randn(self.batch_size, self.latent_dim)
#
#     def init_buffer(self) -> None:
#         n_iter = self.buffer.buffer_size // self.batch_size
#         logger.info(
#             f"Filling buffer of size {self.buffer_size} with {n_iter + 1} iterations"
#         )
#         for i in range(n_iter + 1):
#             logger.info(f"Filling buffer iteration [{i + 1}/{n_iter + 1}]")
#             if self.generator is not None:
#                 x = self.init_sampler().to(self.device)
#                 with torch.no_grad():
#                     x = self.generator(x)
#             else:
#                 x = torch.randn(self.batch_size, self.f_input_dim).to(self.device)
#             values = self.f(x.to(self.f_device))
#             self.buffer.insert_many(list(values), list(x.detach()))
#
#     @abstractmethod
#     def step(self) -> torch.Tensor:
#         pass
#
#     def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
#         pass
