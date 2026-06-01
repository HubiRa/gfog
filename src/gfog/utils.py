import torch
import torch.nn.functional as F


_NEG_INF = -torch.finfo(torch.float32).max


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, neg_inf)
    valid_rows = mask.any(dim=1)
    if not valid_rows.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.logsumexp(masked_logits[valid_rows], dim=1).mean()


def cross_similarity_loss(
    x: torch.Tensor, y: torch.Tensor, temperature: float = 0.7
) -> torch.Tensor:
    """Repulsion loss against another set.

    Penalizes high cosine similarity to *any* member of y instead of assuming an
    arbitrary 1:1 pairing between x[i] and y[i].
    """
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    nx = F.normalize(x, dim=-1)
    ny = F.normalize(y.to(device=x.device, dtype=x.dtype), dim=-1)
    logits_xy = nx @ ny.T / temperature
    logits_yx = ny @ nx.T / temperature
    return 0.5 * (
        torch.logsumexp(logits_xy, dim=1).mean()
        + torch.logsumexp(logits_yx, dim=1).mean()
    )


def self_similarity_loss(x: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    if x.size(0) < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    nx = F.normalize(x, dim=-1)
    logits = nx @ nx.T / temperature
    mask = ~torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    return _masked_logsumexp(logits, mask)


def cross_siglip(
    x: torch.Tensor, y: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Negative-pairs-only SigLIP-style repulsion between two sets."""
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    nx = F.normalize(x, dim=-1)
    ny = F.normalize(y.to(device=x.device, dtype=x.dtype), dim=-1)
    logits = nx @ ny.T / temperature
    return -F.logsigmoid(-logits).mean()


def self_siglip(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if x.size(0) < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    nx = F.normalize(x, dim=-1)
    logits = nx @ nx.T / temperature
    mask = ~torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    return -F.logsigmoid(-logits[mask]).mean()


def uniformity_loss(x: torch.Tensor, t: float = 2.0, eps: float = 1e-8) -> torch.Tensor:
    """Wang–Isola uniformity loss on the hypersphere.

    Computes log E[exp(-t * ||xi - xj||^2)] over pairwise pairs in the batch.
    Returns 0 when batch has fewer than 2 samples.
    """
    if x.size(0) < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    nx = F.normalize(x, dim=-1)
    sq_dists = torch.pdist(nx, p=2).pow(2)
    z = -t * sq_dists
    return torch.logsumexp(z, dim=0) - torch.log(
        torch.tensor(max(z.numel(), 1), device=z.device, dtype=z.dtype)
    )
