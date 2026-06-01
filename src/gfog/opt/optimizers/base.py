import math
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, TypedDict

import torch
from loguru import logger
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, track

from ..components import OptComponents


class PrintTableOptions(TypedDict):
    print_table_every_n_steps: int
    k_best: int


class BaseOpt(ABC):
    def __init__(self, components: OptComponents) -> None:
        self.components = components
        self.fn = self.components.fn
        self.gan = self.components.gan
        self.buffer = self.components.buffer
        self.init_buffer()

    def init_buffer(self) -> None:
        n_iter = math.ceil(self.buffer.B.buffer_size / self.components.batch_size)
        logger.info(
            f"Filling buffer of size {self.buffer.B.buffer_size} with {n_iter} iterations"
        )
        for _ in track(range(n_iter), description="Filling buffer: "):
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
        verbose: bool = False,
        print_table_options: PrintTableOptions | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if "verbous" in kwargs:
            warnings.warn(
                "'verbous' is deprecated; use 'verbose' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            verbose = kwargs.pop("verbous")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

        def take_step() -> bool:
            self.step()
            if termination_eps is not None:
                if (
                    abs(
                        self.buffer.B.get_value(0, level=-1)
                        - self.buffer.B.get_mean_buffer_value(level=-1)
                    )
                    < termination_eps
                ):
                    return True
            return False

        if not verbose:
            for _ in range(n_iter):
                if take_step():
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
                print_every_n_steps = 0
                if print_table_options is not None:
                    print_every_n_steps = print_table_options.get(
                        "print_table_every_n_steps", 0
                    )

                for i in range(n_iter):
                    if take_step():
                        break
                    progress.update(
                        task,
                        advance=1,
                        best=self.buffer.B.get_value(0, level=-1),
                        mean=self.buffer.B.get_mean_buffer_value(level=-1),
                    )

                    if print_every_n_steps > 0 and i % print_every_n_steps == 0:
                        k = (
                            print_table_options.get("k_best", 3)
                            if print_table_options
                            else 3
                        )
                        self.buffer.B.print_values(slice(0, k, 1))

        return self.buffer.B.get_top_k(1)
