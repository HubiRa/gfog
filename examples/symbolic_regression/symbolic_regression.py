"""Symbolic regression with GFog over hard-decoded expression trees.

GFog emits continuous genomes. The objective hard-decodes those genomes into a
fixed-depth expression tree, evaluates the expression on target samples, and
returns MSE plus a small complexity penalty. The operator choices are discrete
inside f, so this is a black-box symbolic-regression task.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels, Rung
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import (
    BaseOpt,
    DefaultOpt,
    LSGANOpt,
    WGANOpt,
    components,
    make_torch_optimizer,
)
from gfog.opt.latents_sampler import LatentSamplerLambda


OPS = ("const", "add", "sub", "mul", "sin", "cos")
OP_ARITY = {"const": 0, "add": 2, "sub": 2, "mul": 2, "sin": 1, "cos": 1}


class SetGenerator(nn.Module):
    """Batch-context generator using self-attention over latent proposals."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        set_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if set_dim % heads != 0:
            raise ValueError(
                f"set_dim must be divisible by heads; got {set_dim=} {heads=}"
            )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, set_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=set_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * set_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(set_dim),
            nn.Linear(set_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_proj(z).unsqueeze(0)).squeeze(0)
        return self.output_head(encoded)


class SetDiscriminator(nn.Module):
    """Contextual per-candidate scorer using self-attention over candidates."""

    def __init__(
        self,
        *,
        input_dim: int,
        set_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if set_dim % heads != 0:
            raise ValueError(
                f"set_dim must be divisible by heads; got {set_dim=} {heads=}"
            )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, set_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=set_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * set_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.score_head = nn.Sequential(
            nn.LayerNorm(set_dim),
            nn.Linear(set_dim, 1),
        )

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_proj(candidates).unsqueeze(0)).squeeze(0)
        return self.score_head(encoded)


@dataclass(frozen=True)
class ExpressionTreeShape:
    depth: int
    n_ops: int = len(OPS)

    @property
    def n_internal(self) -> int:
        return 2**self.depth - 1

    @property
    def n_leaves(self) -> int:
        return 2**self.depth

    @property
    def n_params(self) -> int:
        return self.n_internal * self.n_ops + self.n_leaves * 2


def make_dataset(
    *,
    target: str,
    n_samples: int,
    x_min: float,
    x_max: float,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.linspace(x_min, x_max, n_samples).reshape(-1, 1)
    if target == "nguyen1":
        y = x**3 + x**2 + x
    elif target == "nguyen4":
        y = x**6 + x**5 + x**4 + x**3 + x**2 + x
    elif target == "sin_poly":
        y = torch.sin(x) + x**2
    elif target == "cos_mul":
        y = x * torch.cos(x)
    elif target == "mixed_trig_poly":
        y = x**4 - 0.5 * x**2 + torch.sin(3.0 * x) + 0.5 * x * torch.cos(2.0 * x)
    else:
        raise ValueError(
            "target must be one of nguyen1, nguyen4, sin_poly, cos_mul, mixed_trig_poly; "
            f"got {target}"
        )
    if noise > 0:
        y = y + noise * torch.randn(y.shape, generator=gen)
    return x, y


def safe_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.clamp(left * right, -1e4, 1e4)


def parse_thresholds(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold list must contain at least one value")
    return values


class SymbolicRegressionObjective:
    """Black-box objective over hard-decoded expression-tree genomes."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        shape: ExpressionTreeShape,
        coefficient_scale: float,
        complexity_weight: float,
        unary_weight: float,
        coefficient_l1_weight: float,
        roughness_weight: float,
        lexicographic_simplicity_weight: float,
        ordering: str,
        mse_bucket_size: float,
        refit_steps: int,
        refit_lr: float,
    ) -> None:
        self.x = x.to(torch.float32)
        self.y = y.to(torch.float32)
        self.shape = shape
        self.coefficient_scale = coefficient_scale
        self.complexity_weight = complexity_weight
        self.unary_weight = unary_weight
        self.coefficient_l1_weight = coefficient_l1_weight
        self.roughness_weight = roughness_weight
        self.lexicographic_simplicity_weight = lexicographic_simplicity_weight
        if ordering not in {"scalar", "lexicographic", "cascade"}:
            raise ValueError(
                f"ordering must be scalar, lexicographic, or cascade; got {ordering}"
            )
        self.ordering = ordering
        if mse_bucket_size < 0:
            raise ValueError(f"mse_bucket_size must be >= 0, got {mse_bucket_size}")
        self.mse_bucket_size = mse_bucket_size
        if refit_steps < 0:
            raise ValueError(f"refit_steps must be >= 0, got {refit_steps}")
        if refit_lr <= 0:
            raise ValueError(f"refit_lr must be > 0, got {refit_lr}")
        self.refit_steps = refit_steps
        self.refit_lr = refit_lr

    def _simplicity_penalty(
        self,
        op_ids: torch.Tensor,
        leaf_params: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        binary_ops = torch.zeros_like(op_ids, dtype=torch.float32)
        unary_ops = torch.zeros_like(op_ids, dtype=torch.float32)
        const_ops = torch.zeros_like(op_ids, dtype=torch.float32)
        for op_id, op_name in enumerate(OPS):
            op_mask = (op_ids == op_id).to(torch.float32)
            if OP_ARITY[op_name] == 2:
                binary_ops = binary_ops + op_mask
            elif OP_ARITY[op_name] == 1:
                unary_ops = unary_ops + op_mask
            else:
                const_ops = const_ops + op_mask
        coefficient_l1 = torch.mean(leaf_params.abs(), dim=(1, 2))
        roughness = torch.mean(
            (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]) ** 2, dim=1
        )
        active_structural_ops = binary_ops.sum(dim=1) + unary_ops.sum(dim=1)
        return (
            self.complexity_weight * binary_ops.mean(dim=1)
            + self.unary_weight * unary_ops.mean(dim=1)
            + self.coefficient_l1_weight * coefficient_l1
            + self.roughness_weight * roughness
            + self.lexicographic_simplicity_weight * active_structural_ops
            - self.lexicographic_simplicity_weight * 0.1 * const_ops.sum(dim=1)
        )

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        op_logits_size = self.shape.n_internal * self.shape.n_ops
        op_logits = params[:, :op_logits_size].reshape(
            params.shape[0],
            self.shape.n_internal,
            self.shape.n_ops,
        )
        op_ids = torch.argmax(op_logits, dim=-1)
        leaf_params = self.coefficient_scale * torch.tanh(params[:, op_logits_size:])
        leaf_params = leaf_params.reshape(params.shape[0], self.shape.n_leaves, 2)
        return op_ids, leaf_params

    def _evaluate_decoded(
        self,
        op_ids: torch.Tensor,
        leaf_params: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for leaf_idx in range(self.shape.n_leaves):
            a = leaf_params[:, leaf_idx, 0]
            b = leaf_params[:, leaf_idx, 1]
            values.append(a[:, None, None] * self.x[None, :, :] + b[:, None, None])

        op_offset = self.shape.n_internal - 1
        for level_width in (2**level for level in range(self.shape.depth - 1, -1, -1)):
            next_values = []
            node_offset = op_offset - level_width + 1
            for node_idx in range(level_width):
                left = values[2 * node_idx]
                right = values[2 * node_idx + 1]
                ops = op_ids[:, node_offset + node_idx]
                node_value = torch.zeros_like(left)
                for op_id, op_name in enumerate(OPS):
                    mask = ops == op_id
                    if not torch.any(mask):
                        continue
                    if op_name == "const":
                        out = torch.zeros_like(left[mask])
                    elif op_name == "add":
                        out = left[mask] + right[mask]
                    elif op_name == "sub":
                        out = left[mask] - right[mask]
                    elif op_name == "mul":
                        out = safe_mul(left[mask], right[mask])
                    elif op_name == "sin":
                        out = torch.sin(left[mask])
                    elif op_name == "cos":
                        out = torch.cos(left[mask])
                    else:
                        raise RuntimeError(f"Unhandled op: {op_name}")
                    node_value[mask] = out
                next_values.append(torch.clamp(node_value, -1e4, 1e4))
            values = next_values
            op_offset = node_offset - 1
        return values[0].squeeze(-1)

    def _evaluate_batch(self, params: torch.Tensor) -> torch.Tensor:
        op_ids, leaf_params = self._decode_batch(params)
        return self._evaluate_decoded(op_ids, leaf_params)

    def refit_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Refit affine leaf coefficients while keeping hard operator choices fixed."""
        if self.refit_steps == 0:
            return candidates.detach()
        params = candidates.detach().cpu()
        op_logits_size = self.shape.n_internal * self.shape.n_ops
        op_logits = params[:, :op_logits_size]
        op_ids, _ = self._decode_batch(params)
        raw_leaf_params = params[:, op_logits_size:].reshape(
            params.shape[0],
            self.shape.n_leaves,
            2,
        )
        raw_leaf_params = raw_leaf_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw_leaf_params], lr=self.refit_lr)
        target = self.y.reshape(1, -1)
        with torch.enable_grad():
            for _ in range(self.refit_steps):
                optimizer.zero_grad()
                leaf_params = self.coefficient_scale * torch.tanh(raw_leaf_params)
                pred = self._evaluate_decoded(op_ids, leaf_params)
                loss = torch.mean((pred - target) ** 2)
                if not loss.requires_grad:
                    break
                loss.backward()
                optimizer.step()
        refit_raw = (
            raw_leaf_params.detach().clamp(-8.0, 8.0).reshape(params.shape[0], -1)
        )
        refit_params = torch.cat([op_logits, refit_raw], dim=1)
        return refit_params.to(candidates.device, candidates.dtype)

    def value_from_params(self, params: torch.Tensor) -> torch.Tensor:
        params = params.detach().cpu()
        pred = self._evaluate_batch(params)
        mse = torch.mean((pred - self.y.reshape(1, -1)) ** 2, dim=1)
        op_ids, leaf_params = self._decode_batch(params)
        if self.ordering in {"lexicographic", "cascade"}:
            metrics = self._regularization_metrics_from_decoded(
                params, op_ids, leaf_params, pred
            )
            if self.ordering == "cascade":
                values = torch.stack(
                    [
                        mse,
                        metrics["active_ops"],
                        metrics["coefficient_l1"],
                    ],
                    dim=1,
                )
                return values.to(params.device, params.dtype)
            if self.mse_bucket_size > 0:
                mse_key = torch.floor(mse / self.mse_bucket_size) * self.mse_bucket_size
            else:
                mse_key = mse
            values = torch.stack(
                [
                    mse_key,
                    metrics["active_ops"],
                    metrics["coefficient_l1"],
                    mse,
                ],
                dim=1,
            )
            return values.to(params.device, params.dtype)

        values = mse + self._simplicity_penalty(op_ids, leaf_params, pred)
        return values.to(params.device, params.dtype)

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = self.refit_candidates(candidates)
        values = self.value_from_params(params)
        return values.to(candidates.device, candidates.dtype)

    def _regularization_metrics_from_decoded(
        self,
        params: torch.Tensor,
        op_ids: torch.Tensor,
        leaf_params: torch.Tensor,
        pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del params
        binary_ops = torch.zeros(op_ids.shape[0], dtype=torch.float32)
        unary_ops = torch.zeros(op_ids.shape[0], dtype=torch.float32)
        const_ops = torch.zeros(op_ids.shape[0], dtype=torch.float32)
        for op_id, op_name in enumerate(OPS):
            op_count = (op_ids == op_id).to(torch.float32).sum(dim=1)
            if OP_ARITY[op_name] == 2:
                binary_ops = binary_ops + op_count
            elif OP_ARITY[op_name] == 1:
                unary_ops = unary_ops + op_count
            else:
                const_ops = const_ops + op_count
        roughness = torch.mean(
            (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]) ** 2, dim=1
        )
        return {
            "binary_ops": binary_ops,
            "unary_ops": unary_ops,
            "const_ops": const_ops,
            "active_ops": binary_ops + unary_ops,
            "coefficient_l1": torch.mean(leaf_params.abs(), dim=(1, 2)),
            "roughness": roughness,
        }

    def regularization_metrics(self, params: torch.Tensor) -> dict[str, float]:
        op_ids, leaf_params = self._decode_batch(params.detach().cpu().reshape(1, -1))
        binary_ops = 0
        unary_ops = 0
        const_ops = 0
        for op_id, op_name in enumerate(OPS):
            count = int((op_ids[0] == op_id).sum().item())
            if OP_ARITY[op_name] == 2:
                binary_ops += count
            elif OP_ARITY[op_name] == 1:
                unary_ops += count
            else:
                const_ops += count
        pred = self._evaluate_batch(params.detach().cpu().reshape(1, -1))[0]
        roughness = torch.mean((pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) ** 2)
        return {
            "binary_ops": float(binary_ops),
            "unary_ops": float(unary_ops),
            "const_ops": float(const_ops),
            "active_ops": float(binary_ops + unary_ops),
            "coefficient_l1": float(torch.mean(leaf_params.abs()).item()),
            "roughness": float(roughness.item()),
        }

    def mse(self, params: torch.Tensor) -> float:
        pred = self._evaluate_batch(params.detach().cpu().reshape(1, -1))
        return float(torch.mean((pred[0] - self.y.reshape(-1)) ** 2).item())

    def expression(self, params: torch.Tensor, precision: int = 3) -> str:
        op_ids, leaf_params = self._decode_batch(params.detach().cpu().reshape(1, -1))
        op_ids = op_ids[0].tolist()
        leaf_params = leaf_params[0]
        exprs = []
        for leaf_idx in range(self.shape.n_leaves):
            a = float(leaf_params[leaf_idx, 0].item())
            b = float(leaf_params[leaf_idx, 1].item())
            exprs.append(f"({a:.{precision}f}*x + {b:.{precision}f})")

        op_offset = self.shape.n_internal - 1
        for level_width in (2**level for level in range(self.shape.depth - 1, -1, -1)):
            next_exprs = []
            node_offset = op_offset - level_width + 1
            for node_idx in range(level_width):
                left = exprs[2 * node_idx]
                right = exprs[2 * node_idx + 1]
                op_name = OPS[op_ids[node_offset + node_idx]]
                if op_name == "const":
                    next_exprs.append("0")
                elif op_name == "add":
                    next_exprs.append(f"({left} + {right})")
                elif op_name == "sub":
                    next_exprs.append(f"({left} - {right})")
                elif op_name == "mul":
                    next_exprs.append(f"({left} * {right})")
                elif op_name == "sin":
                    next_exprs.append(f"sin({left})")
                elif op_name == "cos":
                    next_exprs.append(f"cos({left})")
                else:
                    raise RuntimeError(f"Unhandled op: {op_name}")
            exprs = next_exprs
            op_offset = node_offset - 1
        return exprs[0]


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


class RankedOpt(BaseOpt):
    """Rank-target GFog optimizer for scalar black-box objectives."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int,
        ranker_sample_pool_size: int,
        ranker_tau: float,
    ) -> None:
        super().__init__(opt_components)
        self.ranker_list_size = ranker_list_size
        self.ranker_sample_pool_size = ranker_sample_pool_size
        self.ranker_tau = ranker_tau

    def _ranked_buffer_subset(self) -> torch.Tensor:
        current_len = len(self.buffer.B)
        if current_len == 0:
            raise RuntimeError("Cannot sample ranker list from empty buffer")
        pool_size = min(
            current_len, max(self.ranker_list_size, self.ranker_sample_pool_size)
        )
        k = min(self.ranker_list_size, pool_size)
        if k == pool_size:
            ranked = self.buffer.B.get_top_k(k)
        else:
            positions = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.B.get(int(pos)) for pos in positions])
        return ranked.to(self.gan.device, self.gan.dtype)

    def _train_discriminator_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self._ranked_buffer_subset()
        real_scores = self.gan.D(ranked)
        targets = rank_targets(
            real_scores.numel(),
            device=real_scores.device,
            dtype=real_scores.dtype,
            tau=self.ranker_tau,
        ).reshape_as(real_scores)
        real_loss = F.mse_loss(real_scores, targets)
        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            fake = self.gan.G(z)
        fake_scores = self.gan.D(fake.detach())
        fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
        loss = real_loss + fake_loss
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_step()
        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = F.mse_loss(scores, torch.ones_like(scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals

    def evaluate(self, proposals: torch.Tensor) -> None:
        candidates = proposals.detach().to(self.fn.device, self.fn.dtype)
        objective = self.fn.f
        if hasattr(objective, "refit_candidates") and hasattr(
            objective, "value_from_params"
        ):
            evaluated = objective.refit_candidates(candidates)
            values = objective.value_from_params(evaluated).to(
                self.fn.device, self.fn.dtype
            )
        else:
            evaluated = candidates
            values = objective(evaluated)
        self.buffer.B.insert_many(values=list(values), tensors=list(evaluated.detach()))


def build_optimizer(
    args: argparse.Namespace,
) -> tuple[DefaultOpt | LSGANOpt | WGANOpt | RankedOpt, SymbolicRegressionObjective]:
    device = torch.device("cpu")
    shape = ExpressionTreeShape(depth=args.depth)
    x, y = make_dataset(
        target=args.target,
        n_samples=args.n_samples,
        x_min=args.x_min,
        x_max=args.x_max,
        noise=args.noise,
        seed=args.seed,
    )
    objective = SymbolicRegressionObjective(
        x=x,
        y=y,
        shape=shape,
        coefficient_scale=args.coefficient_scale,
        complexity_weight=args.complexity_weight,
        unary_weight=args.unary_weight,
        coefficient_l1_weight=args.coefficient_l1_weight,
        roughness_weight=args.roughness_weight,
        lexicographic_simplicity_weight=args.lexicographic_simplicity_weight,
        ordering=args.ordering,
        mse_bucket_size=args.mse_bucket_size,
        refit_steps=args.refit_steps,
        refit_lr=args.refit_lr,
    )
    fn = components.Fn(
        f=objective,
        input_dim=shape.n_params,
        device=device,
        dtype=torch.float32,
    )
    if args.ordering == "lexicographic":
        value_levels: Levels | int = Levels(
            ["mse_key", "active_ops", "coefficient_l1", "raw_mse"]
        )
    elif args.ordering == "cascade":
        value_levels = Levels.ladder(
            [
                Rung.minimize("mse", parse_thresholds(args.cascade_mse_thresholds)),
                Rung.minimize(
                    "active_ops", parse_thresholds(args.cascade_active_thresholds)
                ),
                Rung.minimize(
                    "coefficient_l1", parse_thresholds(args.cascade_l1_thresholds)
                ),
            ],
            interleave=True,
            final_open="mse",
        )
    else:
        value_levels = 1
    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=value_levels,
        )
    )
    if args.generator_type == "mlp":
        g = MLP(
            input_dim=args.latent_dim,
            output_dim=shape.n_params,
            hidden_dims=[args.generator_hidden_dim, args.generator_hidden_dim],
        )
    elif args.generator_type == "set":
        g = SetGenerator(
            input_dim=args.latent_dim,
            output_dim=shape.n_params,
            set_dim=args.set_dim,
            depth=args.set_depth,
            heads=args.set_heads,
            mlp_ratio=args.set_mlp_ratio,
            dropout=args.set_dropout,
        )
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")

    if args.discriminator_type == "mlp":
        d = MLP(
            input_dim=shape.n_params,
            output_dim=1,
            hidden_dims=[args.discriminator_hidden_dim, args.discriminator_hidden_dim],
        )
    elif args.discriminator_type == "set":
        d = SetDiscriminator(
            input_dim=shape.n_params,
            set_dim=args.set_dim,
            depth=args.set_depth,
            heads=args.set_heads,
            mlp_ratio=args.set_mlp_ratio,
            dropout=args.set_dropout,
        )
    else:
        raise ValueError(f"Unknown discriminator_type: {args.discriminator_type}")
    g = g.to(device)
    d = d.to(device)
    curiosity_loss = None
    if args.uniformity_weight > 0:
        curiosity_loss = WangIsolaUniformity(
            WangIsolaUniformityConfig(
                t=args.uniformity_t,
                use_buffer=args.uniformity_reference == "buffer",
                weight=args.uniformity_weight,
            ),
            buffer=buffer.B,
        )
    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
        latent_dim=args.latent_dim,
        optimizerG=make_torch_optimizer(
            args.g_optimizer,
            g.parameters(),
            lr=args.g_lr,
            momentum=args.optimizer_momentum,
        ),
        optimizerD=make_torch_optimizer(
            args.d_optimizer,
            d.parameters(),
            lr=args.d_lr,
            momentum=args.optimizer_momentum,
        ),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d),
            b=args.batch_size,
            d=args.latent_dim,
        ),
        device=device,
        dtype=torch.float32,
    )
    opt_components = components.OptComponents(
        fn=fn,
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=args.discriminator_steps,
        elite_sampling="random_top_k",
        elite_pool_size=args.buffer_multiplier * args.batch_size,
        weight_clip=args.weight_clip if args.optimizer == "wgan" else None,
    )
    if args.optimizer == "ranked":
        return (
            RankedOpt(
                opt_components,
                ranker_list_size=args.ranker_list_size,
                ranker_sample_pool_size=args.ranker_sample_pool_size,
                ranker_tau=args.ranker_tau,
            ),
            objective,
        )

    opt_cls: type[DefaultOpt | LSGANOpt | WGANOpt]
    if args.optimizer == "default":
        opt_cls = DefaultOpt
    elif args.optimizer == "lsgan":
        opt_cls = LSGANOpt
    elif args.optimizer == "wgan":
        opt_cls = WGANOpt
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    return opt_cls(opt_components), objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["nguyen1", "nguyen4", "sin_poly", "cos_mul", "mixed_trig_poly"],
        default="nguyen1",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--x_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--coefficient_scale", type=float, default=3.0)
    parser.add_argument("--complexity_weight", type=float, default=1e-4)
    parser.add_argument(
        "--unary_weight",
        type=float,
        default=0.0,
        help="Penalty for unary ops such as sin/cos.",
    )
    parser.add_argument(
        "--coefficient_l1_weight",
        type=float,
        default=1e-6,
        help="L1 penalty on affine leaf coefficients.",
    )
    parser.add_argument(
        "--roughness_weight",
        type=float,
        default=0.0,
        help="Penalty on squared second finite differences of expression output.",
    )
    parser.add_argument(
        "--lexicographic_simplicity_weight",
        type=float,
        default=0.0,
        help=(
            "Tiny scalarized tie-breaker for simpler trees. Keep much smaller "
            "than expected MSE differences."
        ),
    )
    parser.add_argument(
        "--ordering",
        choices=["scalar", "lexicographic", "cascade"],
        default="scalar",
        help=(
            "Buffer ordering mode. Lexicographic stores "
            "(mse_bucket, active_ops, coefficient_l1, raw_mse). Cascade uses "
            "Levels.ladder over raw (mse, active_ops, coefficient_l1)."
        ),
    )
    parser.add_argument(
        "--mse_bucket_size",
        type=float,
        default=0.0,
        help=(
            "Bucket size for lexicographic MSE. 0 uses exact raw MSE as the "
            "first key; positive values let simplicity break near-ties."
        ),
    )
    parser.add_argument(
        "--refit_steps",
        type=int,
        default=0,
        help=(
            "Optional black-box local search inside f. Hard operator choices are frozen, "
            "and only affine leaf coefficients are refit before scoring/insertion."
        ),
    )
    parser.add_argument(
        "--refit_lr",
        type=float,
        default=0.05,
        help="Adam learning rate for --refit_steps.",
    )
    parser.add_argument(
        "--cascade_mse_thresholds",
        type=str,
        default="0.1,0.01,0.001",
        help="Comma-separated MSE thresholds for --ordering cascade.",
    )
    parser.add_argument(
        "--cascade_active_thresholds",
        type=str,
        default="7,5,3",
        help="Comma-separated active-op thresholds for --ordering cascade.",
    )
    parser.add_argument(
        "--cascade_l1_thresholds",
        type=str,
        default="1.0,0.5,0.25",
        help="Comma-separated coefficient-L1 thresholds for --ordering cascade.",
    )
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--generator_type", choices=["mlp", "set"], default="mlp")
    parser.add_argument("--discriminator_type", choices=["mlp", "set"], default="mlp")
    parser.add_argument("--generator_hidden_dim", type=int, default=128)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=128)
    parser.add_argument("--set_dim", type=int, default=128)
    parser.add_argument("--set_depth", type=int, default=2)
    parser.add_argument("--set_heads", type=int, default=4)
    parser.add_argument("--set_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_dropout", type=float, default=0.0)
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=0.0,
        help="Wang-Isola uniformity weight added to the generator loss.",
    )
    parser.add_argument("--uniformity_t", type=float, default=2.0)
    parser.add_argument(
        "--uniformity_reference",
        choices=["generated", "buffer"],
        default="buffer",
        help="Use generated samples only, or generated samples plus top buffer elites.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["default", "lsgan", "wgan", "ranked"],
        default="ranked",
    )
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.1)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/symbolic_regression"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer, objective = build_optimizer(args)
    optimizer.optimize(args.n_iter, verbose=True)

    best = optimizer.buffer.B.get_top_k(1).squeeze(0)
    best_value = float(optimizer.buffer.B.get_value(0))
    best_ordering_values = optimizer.buffer.B.get_sorted_values()[0]
    best_mse = objective.mse(best)
    best_expression = objective.expression(best)
    reg_metrics = objective.regularization_metrics(best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"symbolic_regression_{args.target}_seed_{args.seed}.npz",
        best_params=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_mse=np.asarray([best_mse], dtype=np.float32),
        sorted_values=np.asarray(
            optimizer.buffer.B.get_sorted_values(), dtype=np.float32
        ),
        target=np.asarray([args.target]),
        depth=np.asarray([args.depth], dtype=np.int32),
        generator_type=np.asarray([args.generator_type]),
        discriminator_type=np.asarray([args.discriminator_type]),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        uniformity_weight=np.asarray([args.uniformity_weight], dtype=np.float32),
        uniformity_t=np.asarray([args.uniformity_t], dtype=np.float32),
        uniformity_reference=np.asarray([args.uniformity_reference]),
        ordering=np.asarray([args.ordering]),
        mse_bucket_size=np.asarray([args.mse_bucket_size], dtype=np.float32),
        refit_steps=np.asarray([args.refit_steps], dtype=np.int32),
        refit_lr=np.asarray([args.refit_lr], dtype=np.float32),
        best_ordering_values=np.asarray(best_ordering_values, dtype=np.float32),
        expression=np.asarray([best_expression]),
        binary_ops=np.asarray([reg_metrics["binary_ops"]], dtype=np.float32),
        unary_ops=np.asarray([reg_metrics["unary_ops"]], dtype=np.float32),
        const_ops=np.asarray([reg_metrics["const_ops"]], dtype=np.float32),
        active_ops=np.asarray([reg_metrics["active_ops"]], dtype=np.float32),
        coefficient_l1=np.asarray([reg_metrics["coefficient_l1"]], dtype=np.float32),
        roughness=np.asarray([reg_metrics["roughness"]], dtype=np.float32),
    )
    print(f"target={args.target}")
    print(f"n_params={best.numel()}")
    print(f"g_optimizer={args.g_optimizer}")
    print(f"d_optimizer={args.d_optimizer}")
    print(f"g_lr={args.g_lr}")
    print(f"d_lr={args.d_lr}")
    print(f"best_value={best_value:.8f}")
    print(f"ordering={args.ordering}")
    print(f"refit_steps={args.refit_steps}")
    print(f"ordering_values={best_ordering_values}")
    print(f"best_mse={best_mse:.8f}")
    print(
        "regularization="
        f"binary_ops:{reg_metrics['binary_ops']:.0f} "
        f"unary_ops:{reg_metrics['unary_ops']:.0f} "
        f"const_ops:{reg_metrics['const_ops']:.0f} "
        f"active_ops:{reg_metrics['active_ops']:.0f} "
        f"coefficient_l1:{reg_metrics['coefficient_l1']:.6f} "
        f"roughness:{reg_metrics['roughness']:.6f}"
    )
    print(f"best_expression={best_expression}")
    print(
        f"saved={args.output_dir / f'symbolic_regression_{args.target}_seed_{args.seed}.npz'}"
    )


if __name__ == "__main__":
    main()
