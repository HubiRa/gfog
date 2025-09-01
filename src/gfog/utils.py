import torch
import torch.nn.functional as F


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
        torch.tensor(z.numel(), device=z.device, dtype=z.dtype)
    )
