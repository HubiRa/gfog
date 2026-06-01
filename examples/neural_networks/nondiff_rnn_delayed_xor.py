"""Train a vanilla RNN with GFog on a hard delayed-XOR objective.

GFog optimizes flattened RNN parameters. The black box runs a tanh RNN over
fixed sequences and returns hard-threshold 0/1 error on the final prediction.
There is no BPTT signal through the task objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch import nn

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import BaseOpt, DefaultOpt, LSGANOpt, WGANOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


class SetGenerator(nn.Module):
    """Batch-context generator using self-attention over latent tokens."""

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
        tokens = self.input_proj(z).unsqueeze(0)
        encoded = self.encoder(tokens).squeeze(0)
        return self.output_head(encoded)


class SetDiscriminator(nn.Module):
    """Contextual per-candidate scorer using self-attention over a candidate set."""

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
        tokens = self.input_proj(candidates).unsqueeze(0)
        encoded = self.encoder(tokens).squeeze(0)
        return self.score_head(encoded)


class CrossAttentionGenerator(nn.Module):
    """Generator whose latent queries attend to elite-buffer parameter vectors."""

    def __init__(
        self,
        *,
        input_dim: int,
        context_dim: int,
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
        self.query_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, set_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, set_dim),
            nn.GELU(),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=set_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * set_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(set_dim),
            nn.Linear(set_dim, output_dim),
        )
        self.fallback_context = nn.Parameter(torch.zeros(1, context_dim))
        self._context: torch.Tensor | None = None

    def set_context(self, context: torch.Tensor | None) -> None:
        self._context = None if context is None else context.detach()

    def _memory(self, z: torch.Tensor) -> torch.Tensor:
        context = self.fallback_context if self._context is None else self._context
        context = context.to(device=z.device, dtype=z.dtype)
        return self.context_proj(context).unsqueeze(0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(z).unsqueeze(0)
        decoded = self.decoder(queries, self._memory(z)).squeeze(0)
        return self.output_head(decoded)


class CrossAttentionDiscriminator(nn.Module):
    """Per-candidate scorer whose candidate tokens attend to elite-buffer context."""

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
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, set_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, set_dim),
            nn.GELU(),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=set_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * set_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.score_head = nn.Sequential(
            nn.LayerNorm(set_dim),
            nn.Linear(set_dim, 1),
        )
        self._context: torch.Tensor | None = None

    def set_context(self, context: torch.Tensor | None) -> None:
        self._context = None if context is None else context.detach()

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        context = candidates if self._context is None else self._context
        context = context.to(device=candidates.device, dtype=candidates.dtype)
        queries = self.candidate_proj(candidates).unsqueeze(0)
        memory = self.context_proj(context).unsqueeze(0)
        decoded = self.decoder(queries, memory).squeeze(0)
        return self.score_head(decoded)


@dataclass(frozen=True)
class VanillaRNNShape:
    input_dim: int
    hidden_dim: int
    output_dim: int = 1

    @property
    def n_params(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )


class NonDifferentiableDelayedXORObjective:
    """Black-box objective over flattened vanilla-RNN parameters."""

    def __init__(
        self,
        *,
        sequences: torch.Tensor,
        targets: torch.Tensor,
        shape: VanillaRNNShape,
        sample_mode: str = "fixed",
        eval_batch_size: int | None = None,
        score_repeats: int = 1,
        seq_len: int | None = None,
        seed: int = 0,
        weight_scale: float = 1.5,
        weight_mode: str = "continuous",
        quantization_levels: int = 0,
        ternary_threshold: float = 0.25,
        sparse_fraction: float = 0.25,
        recurrent_param: str = "bounded",
        recurrent_radius: float = 1.0,
        l2_weight: float = 1e-4,
    ) -> None:
        self.sequences = sequences.to(torch.float32)
        self.targets = targets.to(torch.float32)
        self.shape = shape
        if sample_mode not in {"fixed", "random"}:
            raise ValueError(
                f"sample_mode must be one of fixed, random; got {sample_mode}"
            )
        self.sample_mode = sample_mode
        self.eval_batch_size = eval_batch_size or sequences.shape[0]
        if score_repeats <= 0:
            raise ValueError(f"score_repeats must be positive, got {score_repeats}")
        self.score_repeats = score_repeats
        self.seq_len = seq_len or sequences.shape[1]
        self.generator = torch.Generator().manual_seed(seed + 1_000_003)
        self.weight_scale = weight_scale
        if weight_mode not in {"continuous", "sign", "ternary", "sparse_topk"}:
            raise ValueError(
                "weight_mode must be one of continuous, sign, ternary, sparse_topk; "
                f"got {weight_mode}"
            )
        self.weight_mode = weight_mode
        self.quantization_levels = quantization_levels
        if ternary_threshold < 0:
            raise ValueError(f"ternary_threshold must be >= 0, got {ternary_threshold}")
        self.ternary_threshold = ternary_threshold
        if not 0 < sparse_fraction <= 1:
            raise ValueError(
                f"sparse_fraction must be in (0, 1], got {sparse_fraction}"
            )
        self.sparse_fraction = sparse_fraction
        if recurrent_param not in {"bounded", "spectral_radius", "fixed_orthogonal"}:
            raise ValueError(
                "recurrent_param must be one of bounded, spectral_radius, fixed_orthogonal; "
                f"got {recurrent_param}"
            )
        if recurrent_radius <= 0:
            raise ValueError(
                f"recurrent_radius must be positive, got {recurrent_radius}"
            )
        self.recurrent_param = recurrent_param
        self.recurrent_radius = recurrent_radius
        fixed_gen = torch.Generator().manual_seed(seed + 17_171)
        q, _r = torch.linalg.qr(
            torch.randn(
                self.shape.hidden_dim, self.shape.hidden_dim, generator=fixed_gen
            )
        )
        self.fixed_w_hh = q.to(torch.float32) * recurrent_radius
        self.l2_weight = l2_weight

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if params.ndim != 2:
            raise ValueError(f"params must be 2D, got shape {tuple(params.shape)}")
        weights = self.weight_scale * torch.tanh(params)
        if self.weight_mode == "sign":
            weights = self.weight_scale * torch.sign(weights)
        elif self.weight_mode == "ternary":
            threshold = self.ternary_threshold * self.weight_scale
            weights = torch.where(
                weights.abs() >= threshold,
                self.weight_scale * torch.sign(weights),
                torch.zeros_like(weights),
            )
        elif self.weight_mode == "sparse_topk":
            k = max(1, int(round(self.sparse_fraction * weights.shape[1])))
            topk_idx = torch.topk(weights.abs(), k=k, dim=1).indices
            mask = torch.zeros_like(weights)
            mask.scatter_(dim=1, index=topk_idx, value=1.0)
            weights = weights * mask
        elif self.quantization_levels > 1:
            levels = self.quantization_levels - 1
            weights = torch.round(
                (weights + self.weight_scale) / (2 * self.weight_scale) * levels
            )
            weights = weights / levels * (2 * self.weight_scale) - self.weight_scale

        offset = 0
        w_ih_size = self.shape.input_dim * self.shape.hidden_dim
        w_ih = weights[:, offset : offset + w_ih_size].reshape(
            params.shape[0],
            self.shape.input_dim,
            self.shape.hidden_dim,
        )
        offset += w_ih_size
        w_hh_size = self.shape.hidden_dim * self.shape.hidden_dim
        w_hh = weights[:, offset : offset + w_hh_size].reshape(
            params.shape[0],
            self.shape.hidden_dim,
            self.shape.hidden_dim,
        )
        if self.recurrent_param == "spectral_radius":
            eigenvalues = torch.linalg.eigvals(w_hh.to(torch.complex64))
            radius = eigenvalues.abs().amax(dim=1).real.clamp_min(1e-6)
            w_hh = w_hh / radius.reshape(-1, 1, 1) * self.recurrent_radius
        elif self.recurrent_param == "fixed_orthogonal":
            w_hh = self.fixed_w_hh.expand(params.shape[0], -1, -1)
        offset += w_hh_size
        b_h = weights[:, offset : offset + self.shape.hidden_dim]
        offset += self.shape.hidden_dim
        w_out_size = self.shape.hidden_dim * self.shape.output_dim
        w_out = weights[:, offset : offset + w_out_size].reshape(
            params.shape[0],
            self.shape.hidden_dim,
            self.shape.output_dim,
        )
        offset += w_out_size
        b_out = weights[:, offset : offset + self.shape.output_dim]
        return weights, w_ih, w_hh, b_h, w_out, b_out

    def _forward_batch(
        self,
        params: torch.Tensor,
        sequences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, w_ih, w_hh, b_h, w_out, b_out = self._decode_batch(params)
        n_candidates = params.shape[0]
        n_sequences = sequences.shape[0]
        hidden = torch.zeros(n_candidates, n_sequences, self.shape.hidden_dim)
        for step in range(sequences.shape[1]):
            x_t = sequences[:, step, :]
            input_term = torch.einsum("si,cih->csh", x_t, w_ih)
            recurrent_term = torch.einsum("csh,chd->csd", hidden, w_hh)
            hidden = torch.tanh(input_term + recurrent_term + b_h[:, None, :])
        logits = (
            torch.einsum("csh,cho->cso", hidden, w_out) + b_out[:, None, :]
        ).squeeze(-1)
        return logits, weights

    def _sample_eval_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sample_mode == "fixed":
            return self.sequences, self.targets
        return make_delayed_xor_dataset_with_generator(
            n_samples=self.eval_batch_size,
            seq_len=self.seq_len,
            generator=self.generator,
        )

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = candidates.detach().cpu()
        errors = []
        weights = None
        for _ in range(self.score_repeats):
            sequences, targets = self._sample_eval_batch()
            logits, weights = self._forward_batch(params, sequences)
            pred = (logits >= 0).to(torch.float32)
            errors.append((pred != targets[None, :]).to(torch.float32).mean(dim=1))
        error = torch.stack(errors).mean(dim=0)
        if weights is None:
            raise RuntimeError("score_repeats produced no evaluations")
        l2 = self.l2_weight * torch.mean(weights * weights, dim=1)
        return (error + l2).to(candidates.device, candidates.dtype)

    def accuracy(self, params: torch.Tensor) -> float:
        logits, _weights = self._forward_batch(
            params.detach().cpu().reshape(1, -1),
            self.sequences,
        )
        pred = (logits >= 0).to(torch.float32)
        return float((pred[0] == self.targets).to(torch.float32).mean().item())


def make_delayed_xor_dataset(
    *,
    n_samples: int,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")
    gen = torch.Generator().manual_seed(seed)
    marker_a = seq_len // 4
    marker_b = 3 * seq_len // 4
    bits = torch.randint(0, 2, (n_samples, seq_len), generator=gen).to(torch.float32)
    sequences = torch.zeros(n_samples, seq_len, 3)
    targets = torch.remainder(bits[:, marker_a] + bits[:, marker_b], 2.0)
    sequences[:, :, 0] = 2.0 * bits - 1.0
    sequences[:, marker_a, 1] = 1.0
    sequences[:, marker_b, 2] = 1.0
    order = torch.randperm(n_samples, generator=gen)
    return sequences[order], targets[order]


def make_delayed_xor_dataset_with_generator(
    *,
    n_samples: int,
    seq_len: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")
    marker_a = seq_len // 4
    marker_b = 3 * seq_len // 4
    bits = torch.randint(0, 2, (n_samples, seq_len), generator=generator).to(
        torch.float32
    )
    sequences = torch.zeros(n_samples, seq_len, 3)
    targets = torch.remainder(bits[:, marker_a] + bits[:, marker_b], 2.0)
    sequences[:, :, 0] = 2.0 * bits - 1.0
    sequences[:, marker_a, 1] = 1.0
    sequences[:, marker_b, 2] = 1.0
    return sequences, targets


def rank_targets(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    tau: float,
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


class RankedOpt(BaseOpt):
    """Small rank-target GFog optimizer for scalar black-box objectives."""

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

    @staticmethod
    def _set_context(model: torch.nn.Module, context: torch.Tensor | None) -> None:
        set_context = getattr(model, "set_context", None)
        if set_context is not None:
            set_context(context)

    def _train_discriminator_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self._ranked_buffer_subset()
        self._set_context(self.gan.D, ranked)
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
            self._set_context(self.gan.G, ranked)
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
        ranked = self._ranked_buffer_subset()
        self._set_context(self.gan.G, ranked)
        self._set_context(self.gan.D, ranked)
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = F.mse_loss(scores, torch.ones_like(scores))
        loss.backward()
        self.gan.optimizerG.step()
        return proposals

    def evaluate(self, proposals: torch.Tensor) -> None:
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))


def build_optimizer(
    args: argparse.Namespace,
) -> tuple[
    DefaultOpt | LSGANOpt | WGANOpt | RankedOpt,
    NonDifferentiableDelayedXORObjective,
]:
    device = torch.device("cpu")
    shape = VanillaRNNShape(input_dim=3, hidden_dim=args.rnn_hidden_dim)
    sequences, targets = make_delayed_xor_dataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    objective = NonDifferentiableDelayedXORObjective(
        sequences=sequences,
        targets=targets,
        shape=shape,
        sample_mode=args.sample_mode,
        eval_batch_size=args.eval_batch_size,
        score_repeats=args.score_repeats,
        seq_len=args.seq_len,
        seed=args.seed,
        weight_scale=args.task_weight_scale,
        weight_mode=args.weight_mode,
        quantization_levels=args.quantization_levels,
        ternary_threshold=args.ternary_threshold,
        sparse_fraction=args.sparse_fraction,
        recurrent_param=args.recurrent_param,
        recurrent_radius=args.recurrent_radius,
        l2_weight=args.l2_weight,
    )
    fn = components.Fn(
        f=objective,
        input_dim=shape.n_params,
        device=device,
        dtype=torch.float32,
    )
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
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
    elif args.generator_type == "cross":
        g = CrossAttentionGenerator(
            input_dim=args.latent_dim,
            context_dim=shape.n_params,
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
    elif args.discriminator_type == "cross":
        d = CrossAttentionDiscriminator(
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
    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=None,
        latent_dim=args.latent_dim,
        optimizerG=torch.optim.Adam(g.parameters(), lr=args.g_lr),
        optimizerD=torch.optim.Adam(d.parameters(), lr=args.d_lr),
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


class TorchVanillaRNN(torch.nn.Module):
    """Plain tanh RNN baseline trained by BPTT."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_ih = torch.nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w_hh = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = torch.nn.Parameter(torch.randn(hidden_dim, 1) * 0.1)
        self.b_out = torch.nn.Parameter(torch.zeros(1))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(
            sequences.shape[0], self.b_h.numel(), device=sequences.device
        )
        for step in range(sequences.shape[1]):
            x_t = sequences[:, step, :]
            hidden = torch.tanh(x_t @ self.w_ih + hidden @ self.w_hh + self.b_h)
        return (hidden @ self.w_out + self.b_out).reshape(-1)


def run_bptt_baseline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    sequences, targets = make_delayed_xor_dataset(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    model = TorchVanillaRNN(input_dim=3, hidden_dim=args.rnn_hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.bptt_lr)
    for _ in range(args.bptt_steps):
        optimizer.zero_grad()
        logits = model(sequences)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        if args.bptt_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.bptt_clip_norm)
        optimizer.step()

    with torch.no_grad():
        logits = model(sequences)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        pred = (logits >= 0).to(torch.float32)
        accuracy = (pred == targets).to(torch.float32).mean()
    print(f"bptt_loss={float(loss.item()):.6f}")
    print(f"bptt_accuracy={float(accuracy.item()):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--generator_type",
        choices=["mlp", "set", "cross"],
        default="mlp",
        help="Proposal generator model class.",
    )
    parser.add_argument(
        "--discriminator_type",
        choices=["mlp", "set", "cross"],
        default="mlp",
        help="Candidate scorer model class.",
    )
    parser.add_argument("--generator_hidden_dim", type=int, default=128)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=128)
    parser.add_argument("--set_dim", type=int, default=128)
    parser.add_argument("--set_depth", type=int, default=2)
    parser.add_argument("--set_heads", type=int, default=4)
    parser.add_argument("--set_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_dropout", type=float, default=0.0)
    parser.add_argument("--rnn_hidden_dim", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument(
        "--sample_mode",
        choices=["fixed", "random"],
        default="fixed",
        help="Score candidates on a fixed dataset or fresh random sequences per evaluation.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Number of random sequences per candidate when --sample_mode random. Defaults to --n_samples.",
    )
    parser.add_argument(
        "--score_repeats",
        type=int,
        default=1,
        help="Average this many independent score batches per candidate evaluation.",
    )
    parser.add_argument("--task_weight_scale", type=float, default=1.5)
    parser.add_argument(
        "--weight_mode",
        choices=["continuous", "sign", "ternary", "sparse_topk"],
        default="continuous",
        help="Black-box RNN weight decode used inside f.",
    )
    parser.add_argument(
        "--quantization_levels",
        type=int,
        default=8,
        help=(
            "Quantize continuous-mode RNN weights to this many levels; "
            "0 disables quantization."
        ),
    )
    parser.add_argument(
        "--ternary_threshold",
        type=float,
        default=0.25,
        help="Zero weights below threshold * task_weight_scale in ternary mode.",
    )
    parser.add_argument(
        "--sparse_fraction",
        type=float,
        default=0.25,
        help="Fraction of weights kept per candidate in sparse_topk mode.",
    )
    parser.add_argument(
        "--recurrent_param",
        choices=["bounded", "spectral_radius", "fixed_orthogonal"],
        default="bounded",
        help="How to parameterize the recurrent matrix W_hh in the black-box decode.",
    )
    parser.add_argument(
        "--recurrent_radius",
        type=float,
        default=1.0,
        help="Target spectral radius for --recurrent_param spectral_radius.",
    )
    parser.add_argument("--l2_weight", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=["default", "lsgan", "wgan", "ranked"],
        default="default",
    )
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=3e-3)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument(
        "--bptt_baseline",
        action="store_true",
        help="Run standard differentiable BPTT baseline instead of GFog.",
    )
    parser.add_argument("--bptt_steps", type=int, default=500)
    parser.add_argument("--bptt_lr", type=float, default=1e-2)
    parser.add_argument("--bptt_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/nondiff_rnn_delayed_xor"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.bptt_baseline:
        run_bptt_baseline(args)
        return

    optimizer, objective = build_optimizer(args)
    optimizer.optimize(args.n_iter, verbose=True)

    best = optimizer.buffer.B.get_top_k(1).squeeze(0)
    best_value = float(optimizer.buffer.B.get_value(0))
    best_accuracy = objective.accuracy(best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"nondiff_rnn_seed_{args.seed}.npz",
        best_params=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_accuracy=np.asarray([best_accuracy], dtype=np.float32),
        sorted_values=np.asarray(
            optimizer.buffer.B.get_sorted_values(), dtype=np.float32
        ),
        generator_type=np.asarray([args.generator_type]),
        discriminator_type=np.asarray([args.discriminator_type]),
        set_dim=np.asarray([args.set_dim], dtype=np.int32),
        set_depth=np.asarray([args.set_depth], dtype=np.int32),
        set_heads=np.asarray([args.set_heads], dtype=np.int32),
        set_mlp_ratio=np.asarray([args.set_mlp_ratio], dtype=np.int32),
        set_dropout=np.asarray([args.set_dropout], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int32),
        rnn_hidden_dim=np.asarray([args.rnn_hidden_dim], dtype=np.int32),
        weight_mode=np.asarray([args.weight_mode]),
        quantization_levels=np.asarray([args.quantization_levels], dtype=np.int32),
        ternary_threshold=np.asarray([args.ternary_threshold], dtype=np.float32),
        sparse_fraction=np.asarray([args.sparse_fraction], dtype=np.float32),
        sample_mode=np.asarray([args.sample_mode]),
        eval_batch_size=np.asarray(
            [args.n_samples if args.eval_batch_size is None else args.eval_batch_size],
            dtype=np.int32,
        ),
        score_repeats=np.asarray([args.score_repeats], dtype=np.int32),
        recurrent_param=np.asarray([args.recurrent_param]),
        recurrent_radius=np.asarray([args.recurrent_radius], dtype=np.float32),
    )
    print(f"best_value={best_value:.6f}")
    print(f"best_accuracy={best_accuracy:.4f}")
    print(f"saved={args.output_dir / f'nondiff_rnn_seed_{args.seed}.npz'}")


if __name__ == "__main__":
    main()
