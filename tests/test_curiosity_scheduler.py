import torch

from gfog.curiosity.scheduler import Scheduler, warmup_cosine, warmup_cosine_annealing
from gfog.utils import (
    cross_similarity_loss,
    self_siglip,
    self_similarity_loss,
    uniformity_loss,
)


def test_scheduler_clamps_after_total_steps() -> None:
    scheduler = Scheduler(lambda step, total: step / total, total_steps=10)
    values = [scheduler.step() for _ in range(13)]
    assert values[-1] == 1.0
    assert values[-2] == 1.0


def test_warmup_cosine_clamps_after_total_steps() -> None:
    assert warmup_cosine(12, 10) == warmup_cosine(10, 10)


def test_warmup_cosine_annealing_handles_zero_cycle_len_cases() -> None:
    value_a = warmup_cosine_annealing(10, 10, cycles=4, warmup_frac=1.0)
    value_b = warmup_cosine_annealing(4, 4, cycles=8)
    assert isinstance(value_a, float)
    assert isinstance(value_b, float)


def test_self_similarity_losses_ignore_trivial_diagonal_and_backprop() -> None:
    x = torch.randn(4, 3, requires_grad=True)
    loss = self_similarity_loss(x)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(loss)

    y = torch.randn(4, 3, requires_grad=True)
    loss_siglip = self_siglip(y)
    loss_siglip.backward()
    assert y.grad is not None
    assert torch.isfinite(loss_siglip)


def test_cross_similarity_is_permutation_invariant_as_a_set_loss() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    y = x[torch.tensor([2, 0, 3, 1])]
    assert torch.isclose(cross_similarity_loss(x, x), cross_similarity_loss(x, y))


def test_uniformity_loss_returns_scalar() -> None:
    x = torch.randn(8, 5)
    loss = uniformity_loss(x)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
