from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .. import components
from .base import BaseOpt


RankerSampleMode = Literal["random_top_pool", "top_k"]
RankTargetCurve = Literal["linear", "exp"]
UtilityLoss = Literal["smooth_l1", "mse"]


def values_to_rows(values: Any, n_rows: int) -> list[list[float]]:
    """Normalize scalar, row-major, or column-major objective values to rows."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise ValueError("Objective values must be batched")

    values_list = list(values)
    if len(values_list) == 0:
        return []
    first = values_list[0]
    if isinstance(first, torch.Tensor) and first.dim() == 0:
        first = float(first.detach().cpu().item())
    scalar_like = not isinstance(first, Iterable) or isinstance(first, (str, bytes))
    if scalar_like:
        if len(values_list) != n_rows:
            raise ValueError(f"Expected {n_rows} scalar values, got {len(values_list)}")
        return [[float(v)] for v in values_list]

    nested = [
        v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else list(v)
        for v in values_list
    ]
    if len(nested) == n_rows:
        return [[float(x) for x in row] for row in nested]
    if all(len(col) == n_rows for col in nested):
        return [[float(x) for x in row] for row in zip(*nested, strict=True)]
    raise ValueError(
        "Could not infer values layout. Expected row-major sample values or "
        "column-major objective levels."
    )


def plackett_luce_loss(scores_best_to_worst: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for a best-to-worst Plackett-Luce list."""
    scores = scores_best_to_worst.reshape(-1)
    if scores.numel() < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    log_denoms = torch.logcumsumexp(scores.flip(0), dim=0).flip(0)
    return -(scores - log_denoms).mean()


def contextual_plackett_luce_generator_loss(
    proposal_scores: torch.Tensor,
    context_scores: torch.Tensor,
) -> torch.Tensor:
    """Make each proposal rank above the evaluated context list under D."""
    proposals = proposal_scores.reshape(-1)
    context = context_scores.reshape(-1).detach()
    if proposals.numel() == 0:
        return torch.zeros(
            (), device=proposal_scores.device, dtype=proposal_scores.dtype
        )
    if context.numel() == 0:
        return -proposals.mean()
    joint_scores = torch.cat(
        [
            proposals[:, None],
            context.to(device=proposals.device, dtype=proposals.dtype)
            .reshape(1, -1)
            .expand(proposals.shape[0], -1),
        ],
        dim=1,
    )
    return -(proposals - torch.logsumexp(joint_scores, dim=1)).mean()


def rank_targets(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    curve: RankTargetCurve = "linear",
    tau: float = 16.0,
) -> torch.Tensor:
    """Dense target values for best-to-worst buffer ranks."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    ranks = torch.arange(n, device=device, dtype=dtype)
    if curve == "linear":
        if n == 1:
            return torch.ones((1,), device=device, dtype=dtype)
        return 1.0 - ranks / float(n - 1)
    if curve == "exp":
        if tau <= 0:
            raise ValueError(f"ranker_tau must be positive, got {tau}")
        return torch.exp(-ranks / tau)
    raise ValueError(f"Unknown rank target curve: {curve}")


class RankedBufferOpt(BaseOpt):
    """Base class for ranker objectives trained on the sorted buffer."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: RankerSampleMode = "random_top_pool",
    ) -> None:
        if ranker_list_size <= 0:
            raise ValueError(
                f"ranker_list_size must be positive, got {ranker_list_size}"
            )
        if ranker_steps <= 0:
            raise ValueError(f"ranker_steps must be positive, got {ranker_steps}")
        if ranker_sample_pool_size is not None and ranker_sample_pool_size <= 0:
            raise ValueError(
                "ranker_sample_pool_size must be positive when set, "
                f"got {ranker_sample_pool_size}"
            )
        if ranker_sample_mode not in {"random_top_pool", "top_k"}:
            raise ValueError(
                "ranker_sample_mode must be one of random_top_pool, top_k; "
                f"got {ranker_sample_mode}"
            )
        self.ranker_list_size = ranker_list_size
        self.ranker_steps = ranker_steps
        self.ranker_sample_pool_size = ranker_sample_pool_size
        self.ranker_sample_mode = ranker_sample_mode
        self._last_evaluated_tensors: torch.Tensor | None = None
        self._last_evaluated_values: list[list[float]] = []
        super().__init__(opt_components)

    def _ranked_buffer_subset_with_positions(
        self,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_len = len(self.buffer.B)
        if current_len == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if self.ranker_sample_mode == "top_k":
            k = min(k, current_len)
            ranked = self.buffer.B.get_top_k(k)
            positions = torch.arange(k, device=self.gan.device, dtype=torch.long)
            return ranked.to(self.gan.device, self.gan.dtype), positions

        pool_size = current_len
        if self.ranker_sample_pool_size is not None:
            pool_size = min(current_len, max(k, self.ranker_sample_pool_size))
        k = min(k, pool_size)
        if k == pool_size:
            ranked = self.buffer.B.get_top_k(k)
            positions = torch.arange(k, device=self.gan.device, dtype=torch.long)
        else:
            position_list = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.B.get(int(pos)) for pos in position_list])
            positions = torch.as_tensor(
                position_list,
                device=self.gan.device,
                dtype=torch.long,
            )
        return ranked.to(self.gan.device, self.gan.dtype), positions

    def _ranked_buffer_subset(self, k: int) -> torch.Tensor:
        ranked, _positions = self._ranked_buffer_subset_with_positions(k)
        return ranked

    def _sample_mixed_evaluated_items(
        self,
        k: int,
    ) -> tuple[torch.Tensor, list[list[float]]]:
        buffer_values = self.buffer.B.get_sorted_values()
        pool_size = len(buffer_values)
        if self.ranker_sample_pool_size is not None:
            pool_size = min(pool_size, max(k, self.ranker_sample_pool_size))
        buffer_k = min(k, pool_size)
        if buffer_k == pool_size:
            buffer_positions = list(range(buffer_k))
        else:
            buffer_positions = (
                torch.randperm(pool_size)[:buffer_k].sort().values.tolist()
            )

        items: list[tuple[list[float], torch.Tensor]] = [
            (
                [float(v) for v in buffer_values[int(pos)]],
                self.buffer.B.get(int(pos)).detach(),
            )
            for pos in buffer_positions
        ]
        if self._last_evaluated_tensors is not None and self._last_evaluated_values:
            items.extend(
                (
                    [float(v) for v in value],
                    tensor.detach().cpu(),
                )
                for value, tensor in zip(
                    self._last_evaluated_values,
                    self._last_evaluated_tensors.detach().cpu(),
                    strict=True,
                )
            )
        items.sort(key=lambda item: tuple(item[0]))
        if len(items) > k:
            selected = torch.randperm(len(items))[:k].sort().values.tolist()
            items = [items[int(idx)] for idx in selected]
        tensors = torch.stack([tensor for _value, tensor in items]).to(
            self.gan.device,
            self.gan.dtype,
        )
        values = [value for value, _tensor in items]
        return tensors, values

    def _evaluate_values(self, proposals: torch.Tensor) -> list[list[float]]:
        raw_values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        return values_to_rows(raw_values, proposals.shape[0])

    def evaluate(self, proposals: torch.Tensor) -> None:
        value_rows = self._evaluate_values(proposals)
        detached = proposals.detach()
        self.buffer.B.insert_many(values=value_rows, tensors=list(detached))
        self._last_evaluated_tensors = detached.cpu()
        self._last_evaluated_values = value_rows


class QuantileRankedDefaultOpt(RankedBufferOpt):
    """Vanilla GAN with dense rank-quantile targets for evaluated buffer samples."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 1.0,
        ranker_target_curve: RankTargetCurve = "linear",
        ranker_tau: float = 16.0,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: RankerSampleMode = "random_top_pool",
        ranker_target_scope: Literal["local", "global"] = "local",
    ) -> None:
        if ranker_weight < 0:
            raise ValueError(f"ranker_weight must be non-negative, got {ranker_weight}")
        if ranker_target_curve not in {"linear", "exp"}:
            raise ValueError(
                "ranker_target_curve must be one of linear, exp; "
                f"got {ranker_target_curve}"
            )
        if ranker_target_scope not in {"local", "global"}:
            raise ValueError(
                "ranker_target_scope must be one of local, global; "
                f"got {ranker_target_scope}"
            )
        self.ranker_weight = ranker_weight
        self.ranker_target_curve = ranker_target_curve
        self.ranker_tau = ranker_tau
        self.ranker_target_scope = ranker_target_scope
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
        )

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked, positions = self._ranked_buffer_subset_with_positions(
            self.ranker_list_size
        )
        real_scores = self.gan.D(ranked)
        if self.ranker_target_scope == "global":
            real_targets = rank_targets(
                len(self.buffer.B),
                device=real_scores.device,
                dtype=real_scores.dtype,
                curve=self.ranker_target_curve,
                tau=self.ranker_tau,
            )[positions].reshape_as(real_scores)
        else:
            real_targets = rank_targets(
                real_scores.numel(),
                device=real_scores.device,
                dtype=real_scores.dtype,
                curve=self.ranker_target_curve,
                tau=self.ranker_tau,
            ).reshape_as(real_scores)
        real_loss = self.gan.loss(real_scores, real_targets)

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            fake = self.gan.G(z)
        fake_scores = self.gan.D(fake.detach())
        fake_loss = self.gan.loss(fake_scores, torch.zeros_like(fake_scores))
        loss = fake_loss + self.ranker_weight * real_loss
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = self.gan.loss(scores, torch.ones_like(scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals


class HybridContextualUtilityRankerOpt(RankedBufferOpt):
    """Contextual PL ranker with local utility calibration."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: RankerSampleMode = "random_top_pool",
        d_score_center_weight: float = 0.0,
        d_score_scale_weight: float = 0.0,
        d_score_target_std: float = 1.0,
        utility_target_scale: float = 100.0,
        utility_loss: UtilityLoss = "smooth_l1",
        utility_weight: float = 0.1,
        generator_utility_weight: float = 0.1,
        utility_clip: float = 3.0,
    ) -> None:
        if d_score_center_weight < 0:
            raise ValueError(
                f"d_score_center_weight must be non-negative, got {d_score_center_weight}"
            )
        if d_score_scale_weight < 0:
            raise ValueError(
                f"d_score_scale_weight must be non-negative, got {d_score_scale_weight}"
            )
        if d_score_target_std <= 0:
            raise ValueError(
                f"d_score_target_std must be positive, got {d_score_target_std}"
            )
        if utility_target_scale <= 0:
            raise ValueError(
                f"utility_target_scale must be positive, got {utility_target_scale}"
            )
        if utility_loss not in {"smooth_l1", "mse"}:
            raise ValueError(
                f"utility_loss must be one of smooth_l1, mse; got {utility_loss}"
            )
        if utility_weight < 0:
            raise ValueError(
                f"utility_weight must be non-negative, got {utility_weight}"
            )
        if generator_utility_weight < 0:
            raise ValueError(
                "generator_utility_weight must be non-negative, "
                f"got {generator_utility_weight}"
            )
        if utility_clip <= 0:
            raise ValueError(f"utility_clip must be positive, got {utility_clip}")
        self.d_score_center_weight = d_score_center_weight
        self.d_score_scale_weight = d_score_scale_weight
        self.d_score_target_std = d_score_target_std
        self.utility_target_scale = utility_target_scale
        self.utility_loss = utility_loss
        self.utility_weight = utility_weight
        self.generator_utility_weight = generator_utility_weight
        self.utility_clip = utility_clip
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
        )

    def _local_utility_targets(
        self,
        values: list[list[float]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values_array = np.asarray(values, dtype=np.float32)
        if values_array.ndim == 1:
            values_array = values_array[:, None]
        objective = values_array[:, -1]
        reference = float(np.mean(objective))
        scale = float(np.std(objective))
        if scale < 1e-6:
            scale = self.utility_target_scale
        utilities = (reference - objective) / max(scale, 1e-6)
        utilities = np.clip(utilities, -self.utility_clip, self.utility_clip)
        return torch.as_tensor(utilities, device=device, dtype=dtype)

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        tensors, values = self._sample_mixed_evaluated_items(self.ranker_list_size)
        scores = self.gan.D(tensors).reshape(-1)
        pl_loss = plackett_luce_loss(scores)
        targets = self._local_utility_targets(
            values,
            device=scores.device,
            dtype=scores.dtype,
        )
        if self.utility_loss == "mse":
            utility_loss = F.mse_loss(scores, targets)
        else:
            utility_loss = F.smooth_l1_loss(scores, targets)
        loss = pl_loss + self.utility_weight * utility_loss
        if self.d_score_center_weight > 0:
            loss = loss + self.d_score_center_weight * scores.mean().square()
        if self.d_score_scale_weight > 0 and scores.numel() > 1:
            std = scores.std(unbiased=False)
            loss = (
                loss
                + self.d_score_scale_weight * (std - self.d_score_target_std).square()
            )
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        proposal_scores = self.gan.D(proposals).reshape(-1)
        context = self._ranked_buffer_subset(
            min(self.ranker_list_size, len(self.buffer.B))
        )
        with torch.no_grad():
            context_scores = self.gan.D(context).reshape(-1)
        loss = contextual_plackett_luce_generator_loss(
            proposal_scores,
            context_scores,
        )
        loss = loss - self.generator_utility_weight * proposal_scores.mean()
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals
