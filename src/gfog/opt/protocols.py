from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Minimal public optimizer protocol for GFog-compatible optimizers.

    External optimizers do not need to subclass :class:`BaseOpt` as long as they
    provide the same surface API.
    """

    def step(self) -> None: ...

    def optimize(
        self,
        n_iter: int,
        termination_eps: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor: ...
