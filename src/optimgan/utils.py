import torch
import torch.nn as nn
from torch import autograd

from functorch import jacrev

from typing import Callable


def maybe_unsqueeze(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    If x has fewer dimensions than target_dim, unsqueeze it to target_dim.
    """
    if x.dim() < target_dim:
        diff = target_dim - x.dim()
        unsqueeze = [None] * diff
        return x[None]
    elif x.dim() == target_dim:
        return x
    else:
        raise ValueError(
            f"Input tensor has {x.dim()} dimensions, but expected dim <= {target_dim}"
        )


def gradient_penalty(
    out_model: torch.Tensor,
    out_buffer: torch.Tensor,
    batch_size: int,
    D: nn.Module,
    G: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    # WGAN gradient penalty
    epsilon = torch.random.uniform(0, 1, size=(batch_size, 1))
    interpol = epsilon * out_buffer + ((1 - epsilon) * out_model)
    interpol_out = D(interpol)
    grads = autograd.grad(
        outputs=interpol_out,
        inputs=interpol,
        grad_outputs=torch.ones(interpol_out.shape).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.reshape([out_model.shape, -1])
    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return grad_penalty


def generate_weights(f_weight_gen: Callable, state: torch.Tensor) -> torch.Tensor:
    """
    Generate weights for nerual network based on the state.
    """
    w = jacrev(f_weight_gen)(state)
    return w


class WeightGenerator:
    def __init__(self, out_dims: list) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        pass
