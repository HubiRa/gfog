import torch


def maybe_unsqueeze(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    '''
    If x has fewer dimensions than target_dim, unsqueeze it to target_dim.
    '''
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