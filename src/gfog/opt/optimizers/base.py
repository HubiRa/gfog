import torch
from typing import Iterable

from loguru import logger
from abc import ABC, abstractmethod
from rich.progress import track

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
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
        for _ in track(range(n_iter + 1), description="Filling buffer: "):
            if self.gan.G is not None:
                x = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
                with torch.no_grad():
                    x = self.gan.G(x)
            else:
                x = torch.randn(self.components.batch_size, self.fn.input_dim)

            values = self.fn.f(x.to(self.fn.device, self.fn.dtype))
            if len(values) > 0 and not isinstance(values[0], Iterable):
                values = list(values)

            self.buffer.B.insert_many(values=values, tensors=list(x.detach()))

    @abstractmethod
    def propose(self) -> torch.Tensor: ...

    @abstractmethod
    def evaluate(self, proposals: torch.Tensor) -> None: ...

    def step(self) -> None:
        proposals = self.propose()
        self.evaluate(proposals)

    def optimize(
        self,
        n_iter: int,
        termination_eps: float | None = None,
        verbous: bool = False,
    ) -> torch.Tensor:
        def take_step() -> bool:
            self.step()
            if termination_eps is not None:
                if (
                    abs(self.buffer.B.values[0] - self.buffer.B.get_mean_buffer_value())
                    < termination_eps
                ):
                    return True  # stop
            return False  # do not stop

        if not verbous:
            for _ in range(n_iter):
                if stop := take_step():
                    break
        else:
            progress = Progress(
                TextColumn("Iteration {task.completed}"),
                BarColumn(),
                TextColumn("Best: {task.fields[best]:.4f}"),
                TextColumn("Mean: {task.fields[mean]:.4f}"),
                TimeElapsedColumn(),
            )
            with progress:
                task = progress.add_task(
                    "Optimizing", total=n_iter, best=999.0, mean=999.0
                )
                for _ in range(n_iter):
                    if stop := take_step():
                        break
                    progress.update(
                        task,
                        advance=1,
                        best=self.buffer.B.get_value(0, level=-1),
                        mean=self.buffer.B.get_mean_buffer_value(level=-1),
                    )
