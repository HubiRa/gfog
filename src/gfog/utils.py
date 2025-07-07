import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F


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


def cross_similarity_loss(
    x: torch.Tensor, y: torch.Tensor, temperature: float = 0.7
) -> torch.Tensor:
    nx = F.normalize(x, dim=-1)
    ny = F.normalize(y, dim=-1)

    logits1 = torch.einsum("ij,jk->ik", nx, ny.T) / temperature
    logits2 = torch.einsum("ij,jk->ik", ny, nx.T) / temperature

    targets = torch.arange(logits1.shape[0]).to(device=x.device)
    l1 = F.cross_entropy(logits1, target=targets)
    l2 = F.cross_entropy(logits2, target=targets)

    return 0.5 * (l1 + l2)


def self_similarity_loss(x: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    np1 = F.normalize(x, dim=-1)
    logits1 = torch.einsum("ij,jk->ik", np1, np1.T) / temperature
    targets = torch.arange(logits1.shape[0]).to(device=x.device)
    return F.cross_entropy(logits1, target=targets)


def cross_siglip(
    x: torch.Tensor, y: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    nx = F.normalize(x, dim=-1)
    ny = F.normalize(y, dim=-1)

    logits = nx @ ny.T / temperature

    dim = logits.size(0)
    labels = torch.eye(dim, device=logits.device, dtype=logits.dtype)
    labels = 2 * labels - 1

    loss = F.logsigmoid(labels * logits)

    return -loss.mean()


def self_siglip(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    nx = F.normalize(x, dim=-1)

    logits = nx @ nx.T / temperature

    dim = logits.size(0)
    labels = torch.eye(dim, device=logits.device, dtype=logits.dtype)
    labels = 2 * labels - 1

    loss = F.logsigmoid(labels * logits)

    return -loss.mean()
