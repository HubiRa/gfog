from collections.abc import Iterable
from typing import Any

import torch


def zeropower_via_newton_schulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the zeroth power/orthogonal factor of a 2D gradient."""
    if g.ndim != 2:
        raise ValueError(
            f"Muon zeropower expects a 2D tensor, got shape={tuple(g.shape)}"
        )

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T

    x = x / torch.clamp(x.norm(), min=1e-12)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transpose:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Small experimental Muon optimizer for matrix-heavy networks."""

    def __init__(
        self,
        parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                update = grad.add(buf, alpha=beta) if nesterov else buf
                if update.ndim == 2:
                    update_2d = zeropower_via_newton_schulz5(update, steps=ns_steps)
                    scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                    p.add_(update_2d, alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr)
        return loss


def make_torch_optimizer(
    name: str,
    parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    *,
    lr: float,
    momentum: float = 0.95,
) -> torch.optim.Optimizer:
    """Build the standard torch optimizer used by GFog examples."""
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if name == "muon":
        return Muon(parameters, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")
