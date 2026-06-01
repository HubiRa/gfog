"""Optimize binary-weight classifiers on a small image-like digits task.

The generated genome is continuous, but the black-box objective decodes it into
a classifier, optionally sign-binarizes the weights, and returns hard 0/1
multiclass error. G and D remain differentiable; only f is black-box.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import BaseOpt, DefaultOpt, LSGANOpt, WGANOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


@dataclass(frozen=True)
class DigitMLPShape:
    input_dim: int
    hidden_dim: int
    output_dim: int

    @property
    def n_params(self) -> int:
        return (
            self.input_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.output_dim
            + self.output_dim
        )


@dataclass(frozen=True)
class CirculantResidualShape:
    input_dim: int
    width: int
    depth: int
    output_dim: int

    @property
    def n_params(self) -> int:
        input_head = self.input_dim * self.width + self.width
        layers = self.depth * (self.width + self.width)
        output_head = self.width * self.output_dim + self.output_dim
        return input_head + layers + output_head


@dataclass(frozen=True)
class BinaryConvShape:
    channels: int
    depth: int
    output_dim: int
    kernel_size: int = 3

    @property
    def n_params(self) -> int:
        first = self.channels * self.kernel_size * self.kernel_size + self.channels
        hidden = (self.depth - 1) * (
            self.channels * self.channels * self.kernel_size * self.kernel_size
            + self.channels
        )
        output = self.channels * self.output_dim + self.output_dim
        return first + hidden + output


def _digit_templates() -> torch.Tensor:
    raw = [
        [
            "111111",
            "100001",
            "100011",
            "100101",
            "101001",
            "110001",
            "100001",
            "111111",
        ],
        [
            "001100",
            "011100",
            "001100",
            "001100",
            "001100",
            "001100",
            "001100",
            "111111",
        ],
        [
            "111110",
            "000001",
            "000001",
            "111110",
            "100000",
            "100000",
            "100000",
            "111111",
        ],
        [
            "111110",
            "000001",
            "000001",
            "011110",
            "000001",
            "000001",
            "000001",
            "111110",
        ],
        [
            "100010",
            "100010",
            "100010",
            "111111",
            "000010",
            "000010",
            "000010",
            "000010",
        ],
        [
            "111111",
            "100000",
            "100000",
            "111110",
            "000001",
            "000001",
            "000001",
            "111110",
        ],
        [
            "111111",
            "100000",
            "100000",
            "111110",
            "100001",
            "100001",
            "100001",
            "111110",
        ],
        [
            "111111",
            "000001",
            "000010",
            "000100",
            "001000",
            "010000",
            "010000",
            "010000",
        ],
        [
            "111110",
            "100001",
            "100001",
            "111110",
            "100001",
            "100001",
            "100001",
            "111110",
        ],
        [
            "111110",
            "100001",
            "100001",
            "100001",
            "111111",
            "000001",
            "000001",
            "111111",
        ],
    ]
    templates = torch.zeros(10, 8, 8)
    for digit, rows in enumerate(raw):
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                templates[digit, row_idx, col_idx + 1] = float(value)
    return templates


def make_template_digits_dataset(
    *,
    n_samples: int,
    noise: float,
    dropout: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy 8x8 digit-like images without external data dependencies."""
    if n_samples < 10:
        raise ValueError(f"n_samples must be >= 10, got {n_samples}")
    gen = torch.Generator().manual_seed(seed)
    templates = _digit_templates()
    labels = torch.arange(n_samples) % 10
    labels = labels[torch.randperm(n_samples, generator=gen)]
    images = templates[labels].clone()

    shifts = torch.randint(-1, 2, (n_samples, 2), generator=gen)
    shifted = torch.zeros_like(images)
    for idx, (dy, dx) in enumerate(shifts.tolist()):
        shifted[idx] = torch.roll(images[idx], shifts=(dy, dx), dims=(0, 1))
    images = shifted

    if dropout > 0:
        keep = torch.rand(images.shape, generator=gen) >= dropout
        images = images * keep.to(images.dtype)
    if noise > 0:
        images = images + noise * torch.randn(images.shape, generator=gen)
    images = images.clamp(0.0, 1.0)
    images = 2.0 * images - 1.0
    return images.reshape(n_samples, -1), labels


class NonDifferentiableBinaryDigitsObjective:
    """Black-box hard-accuracy objective for generated classifier weights."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        shape: DigitMLPShape,
        weight_scale: float,
        weight_mode: str,
        ternary_threshold: float,
        l2_weight: float,
    ) -> None:
        self.x = x.to(torch.float32)
        self.y = y.to(torch.long)
        self.shape = shape
        self.weight_scale = weight_scale
        if weight_mode not in {"continuous", "sign", "ternary"}:
            raise ValueError(
                f"weight_mode must be one of continuous, sign, ternary; got {weight_mode}"
            )
        self.weight_mode = weight_mode
        self.ternary_threshold = ternary_threshold
        self.l2_weight = l2_weight

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        weights = self.weight_scale * torch.tanh(params)
        if self.weight_mode == "sign":
            weights = self.weight_scale * torch.where(
                weights >= 0,
                torch.ones_like(weights),
                -torch.ones_like(weights),
            )
        elif self.weight_mode == "ternary":
            threshold = self.ternary_threshold * self.weight_scale
            weights = torch.where(
                weights.abs() >= threshold,
                self.weight_scale * torch.sign(weights),
                torch.zeros_like(weights),
            )

        offset = 0
        w1_size = self.shape.input_dim * self.shape.hidden_dim
        w1 = weights[:, offset : offset + w1_size].reshape(
            params.shape[0],
            self.shape.input_dim,
            self.shape.hidden_dim,
        )
        offset += w1_size
        b1 = weights[:, offset : offset + self.shape.hidden_dim]
        offset += self.shape.hidden_dim
        w2_size = self.shape.hidden_dim * self.shape.output_dim
        w2 = weights[:, offset : offset + w2_size].reshape(
            params.shape[0],
            self.shape.hidden_dim,
            self.shape.output_dim,
        )
        offset += w2_size
        b2 = weights[:, offset : offset + self.shape.output_dim]
        return weights, w1, b1, w2, b2

    def _forward_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, w1, b1, w2, b2 = self._decode_batch(params)
        hidden = torch.tanh(torch.einsum("ni,cih->cnh", self.x, w1) + b1[:, None, :])
        logits = torch.einsum("cnh,cho->cno", hidden, w2) + b2[:, None, :]
        return logits, weights

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = candidates.detach().cpu()
        logits, weights = self._forward_batch(params)
        pred = torch.argmax(logits, dim=-1)
        error = (pred != self.y[None, :]).to(torch.float32).mean(dim=1)
        l2 = self.l2_weight * torch.mean(weights * weights, dim=1)
        return (error + l2).to(candidates.device, candidates.dtype)

    def accuracy(self, params: torch.Tensor) -> float:
        logits, _weights = self._forward_batch(params.detach().cpu().reshape(1, -1))
        pred = torch.argmax(logits[0], dim=-1)
        return float((pred == self.y).to(torch.float32).mean().item())


class NonDifferentiableCirculantDigitsObjective:
    """Deep binary residual classifier with circulant hidden layers inside f."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        shape: CirculantResidualShape,
        weight_scale: float,
        weight_mode: str,
        ternary_threshold: float,
        residual_scale: float,
        l2_weight: float,
    ) -> None:
        self.x = x.to(torch.float32)
        self.y = y.to(torch.long)
        self.shape = shape
        self.weight_scale = weight_scale
        if weight_mode not in {"continuous", "sign", "ternary"}:
            raise ValueError(
                f"weight_mode must be one of continuous, sign, ternary; got {weight_mode}"
            )
        self.weight_mode = weight_mode
        self.ternary_threshold = ternary_threshold
        self.residual_scale = residual_scale
        self.l2_weight = l2_weight

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        weights = self.weight_scale * torch.tanh(params)
        if self.weight_mode == "sign":
            weights = self.weight_scale * torch.where(
                weights >= 0,
                torch.ones_like(weights),
                -torch.ones_like(weights),
            )
        elif self.weight_mode == "ternary":
            threshold = self.ternary_threshold * self.weight_scale
            weights = torch.where(
                weights.abs() >= threshold,
                self.weight_scale * torch.sign(weights),
                torch.zeros_like(weights),
            )

        offset = 0
        w_in_size = self.shape.input_dim * self.shape.width
        w_in = weights[:, offset : offset + w_in_size].reshape(
            params.shape[0],
            self.shape.input_dim,
            self.shape.width,
        )
        offset += w_in_size
        b_in = weights[:, offset : offset + self.shape.width]
        offset += self.shape.width
        layer_size = self.shape.depth * self.shape.width
        layer_vecs = weights[:, offset : offset + layer_size].reshape(
            params.shape[0],
            self.shape.depth,
            self.shape.width,
        )
        offset += layer_size
        layer_bias = weights[:, offset : offset + layer_size].reshape(
            params.shape[0],
            self.shape.depth,
            self.shape.width,
        )
        offset += layer_size
        w_out_size = self.shape.width * self.shape.output_dim
        w_out = weights[:, offset : offset + w_out_size].reshape(
            params.shape[0],
            self.shape.width,
            self.shape.output_dim,
        )
        offset += w_out_size
        b_out = weights[:, offset : offset + self.shape.output_dim]
        return weights, w_in, b_in, layer_vecs, layer_bias, w_out, b_out

    def _circulant_mix(
        self, hidden: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        hidden_fft = torch.fft.rfft(hidden, dim=-1)
        kernel_fft = torch.fft.rfft(kernel, dim=-1)
        mixed = torch.fft.irfft(
            hidden_fft * kernel_fft[:, None, :], n=self.shape.width, dim=-1
        )
        return mixed / (self.shape.width**0.5)

    def _forward_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, w_in, b_in, layer_vecs, layer_bias, w_out, b_out = self._decode_batch(
            params
        )
        hidden = torch.tanh(
            torch.einsum("ni,cih->cnh", self.x, w_in) + b_in[:, None, :]
        )
        step_scale = self.residual_scale / max(1, self.shape.depth)
        for layer_idx in range(self.shape.depth):
            mixed = self._circulant_mix(hidden, layer_vecs[:, layer_idx, :])
            update = torch.tanh(mixed + layer_bias[:, layer_idx, :][:, None, :])
            hidden = hidden + step_scale * update
        logits = torch.einsum("cnh,cho->cno", hidden, w_out) + b_out[:, None, :]
        return logits, weights

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = candidates.detach().cpu()
        logits, weights = self._forward_batch(params)
        pred = torch.argmax(logits, dim=-1)
        error = (pred != self.y[None, :]).to(torch.float32).mean(dim=1)
        l2 = self.l2_weight * torch.mean(weights * weights, dim=1)
        return (error + l2).to(candidates.device, candidates.dtype)

    def accuracy(self, params: torch.Tensor) -> float:
        logits, _weights = self._forward_batch(params.detach().cpu().reshape(1, -1))
        pred = torch.argmax(logits[0], dim=-1)
        return float((pred == self.y).to(torch.float32).mean().item())


class NonDifferentiableBinaryConvDigitsObjective:
    """Black-box hard-accuracy objective for tiny binary convolutional nets."""

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        shape: BinaryConvShape,
        weight_scale: float,
        weight_mode: str,
        ternary_threshold: float,
        residual_scale: float,
        l2_weight: float,
    ) -> None:
        self.x = x.to(torch.float32).reshape(-1, 1, 8, 8)
        self.y = y.to(torch.long)
        self.shape = shape
        self.weight_scale = weight_scale
        if weight_mode not in {"continuous", "sign", "ternary"}:
            raise ValueError(
                f"weight_mode must be one of continuous, sign, ternary; got {weight_mode}"
            )
        if shape.depth <= 0:
            raise ValueError(f"conv depth must be positive, got {shape.depth}")
        self.weight_mode = weight_mode
        self.ternary_threshold = ternary_threshold
        self.residual_scale = residual_scale
        self.l2_weight = l2_weight

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        weights = self.weight_scale * torch.tanh(params)
        if self.weight_mode == "sign":
            weights = self.weight_scale * torch.where(
                weights >= 0,
                torch.ones_like(weights),
                -torch.ones_like(weights),
            )
        elif self.weight_mode == "ternary":
            threshold = self.ternary_threshold * self.weight_scale
            weights = torch.where(
                weights.abs() >= threshold,
                self.weight_scale * torch.sign(weights),
                torch.zeros_like(weights),
            )

        offset = 0
        k = self.shape.kernel_size
        first_size = self.shape.channels * k * k
        first_w = weights[:, offset : offset + first_size].reshape(
            params.shape[0],
            self.shape.channels,
            1,
            k,
            k,
        )
        offset += first_size
        first_b = weights[:, offset : offset + self.shape.channels]
        offset += self.shape.channels

        hidden_w = []
        hidden_b = []
        hidden_size = self.shape.channels * self.shape.channels * k * k
        for _ in range(self.shape.depth - 1):
            layer_w = weights[:, offset : offset + hidden_size].reshape(
                params.shape[0],
                self.shape.channels,
                self.shape.channels,
                k,
                k,
            )
            offset += hidden_size
            layer_b = weights[:, offset : offset + self.shape.channels]
            offset += self.shape.channels
            hidden_w.append(layer_w)
            hidden_b.append(layer_b)

        out_size = self.shape.channels * self.shape.output_dim
        out_w = weights[:, offset : offset + out_size].reshape(
            params.shape[0],
            self.shape.channels,
            self.shape.output_dim,
        )
        offset += out_size
        out_b = weights[:, offset : offset + self.shape.output_dim]
        return weights, first_w, first_b, hidden_w, hidden_b, out_w, out_b

    def _conv2d_per_candidate(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        n_candidates = weight.shape[0]
        if x.ndim == 4:
            x = x.unsqueeze(0).expand(n_candidates, -1, -1, -1, -1)
        batch_size = x.shape[1]
        k = self.shape.kernel_size
        patches = F.unfold(
            x.reshape(n_candidates * batch_size, x.shape[2], x.shape[3], x.shape[4]),
            kernel_size=k,
            padding=k // 2,
        )
        patches = patches.reshape(n_candidates, batch_size, x.shape[2] * k * k, -1)
        flat_weight = weight.reshape(n_candidates, weight.shape[1], -1)
        out = torch.einsum("cbpl,cop->cbol", patches, flat_weight)
        out = out + bias[:, None, :, None]
        side = int(out.shape[-1] ** 0.5)
        return out.reshape(n_candidates, batch_size, weight.shape[1], side, side)

    def _forward_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, first_w, first_b, hidden_w, hidden_b, out_w, out_b = (
            self._decode_batch(params)
        )
        hidden = torch.tanh(self._conv2d_per_candidate(self.x, first_w, first_b))
        step_scale = self.residual_scale / max(1, self.shape.depth - 1)
        for layer_w, layer_b in zip(hidden_w, hidden_b, strict=True):
            update = torch.tanh(self._conv2d_per_candidate(hidden, layer_w, layer_b))
            hidden = hidden + step_scale * update
        pooled = hidden.mean(dim=(-1, -2))
        logits = torch.einsum("cnh,cho->cno", pooled, out_w) + out_b[:, None, :]
        return logits, weights

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = candidates.detach().cpu()
        logits, weights = self._forward_batch(params)
        pred = torch.argmax(logits, dim=-1)
        error = (pred != self.y[None, :]).to(torch.float32).mean(dim=1)
        l2 = self.l2_weight * torch.mean(weights * weights, dim=1)
        return (error + l2).to(candidates.device, candidates.dtype)

    def accuracy(self, params: torch.Tensor) -> float:
        logits, _weights = self._forward_batch(params.detach().cpu().reshape(1, -1))
        pred = torch.argmax(logits[0], dim=-1)
        return float((pred == self.y).to(torch.float32).mean().item())


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


class RankedOpt(BaseOpt):
    """Rank-target optimizer for scalar black-box objectives."""

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
    NonDifferentiableBinaryDigitsObjective
    | NonDifferentiableCirculantDigitsObjective
    | NonDifferentiableBinaryConvDigitsObjective,
]:
    device = torch.device("cpu")
    x, y = make_template_digits_dataset(
        n_samples=args.n_samples,
        noise=args.noise,
        dropout=args.dropout,
        seed=args.seed,
    )
    if args.task_arch == "dense_mlp":
        shape = DigitMLPShape(
            input_dim=64, hidden_dim=args.task_hidden_dim, output_dim=10
        )
        objective = NonDifferentiableBinaryDigitsObjective(
            x=x,
            y=y,
            shape=shape,
            weight_scale=args.task_weight_scale,
            weight_mode=args.weight_mode,
            ternary_threshold=args.ternary_threshold,
            l2_weight=args.l2_weight,
        )
    elif args.task_arch == "circulant_residual":
        shape = CirculantResidualShape(
            input_dim=64,
            width=args.circulant_width,
            depth=args.circulant_depth,
            output_dim=10,
        )
        objective = NonDifferentiableCirculantDigitsObjective(
            x=x,
            y=y,
            shape=shape,
            weight_scale=args.task_weight_scale,
            weight_mode=args.weight_mode,
            ternary_threshold=args.ternary_threshold,
            residual_scale=args.residual_scale,
            l2_weight=args.l2_weight,
        )
    elif args.task_arch == "binary_conv":
        shape = BinaryConvShape(
            channels=args.conv_channels,
            depth=args.conv_depth,
            output_dim=10,
        )
        objective = NonDifferentiableBinaryConvDigitsObjective(
            x=x,
            y=y,
            shape=shape,
            weight_scale=args.task_weight_scale,
            weight_mode=args.weight_mode,
            ternary_threshold=args.ternary_threshold,
            residual_scale=args.residual_scale,
            l2_weight=args.l2_weight,
        )
    else:
        raise ValueError(f"Unknown task_arch: {args.task_arch}")
    fn = components.Fn(
        f=objective,
        input_dim=shape.n_params,
        device=device,
        dtype=torch.float32,
    )
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
    )
    g = MLP(
        input_dim=args.latent_dim,
        output_dim=shape.n_params,
        hidden_dims=[args.generator_hidden_dim, args.generator_hidden_dim],
    ).to(device)
    d = MLP(
        input_dim=shape.n_params,
        output_dim=1,
        hidden_dims=[args.discriminator_hidden_dim, args.discriminator_hidden_dim],
    ).to(device)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--generator_hidden_dim", type=int, default=256)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--task_arch",
        choices=["dense_mlp", "circulant_residual", "binary_conv"],
        default="dense_mlp",
    )
    parser.add_argument("--task_hidden_dim", type=int, default=64)
    parser.add_argument("--circulant_width", type=int, default=64)
    parser.add_argument("--circulant_depth", type=int, default=16)
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
        help="Total residual update scale distributed as residual_scale / depth.",
    )
    parser.add_argument("--conv_channels", type=int, default=8)
    parser.add_argument("--conv_depth", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--noise", type=float, default=0.20)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--task_weight_scale", type=float, default=1.0)
    parser.add_argument(
        "--weight_mode",
        choices=["continuous", "sign", "ternary"],
        default="sign",
    )
    parser.add_argument("--ternary_threshold", type=float, default=0.25)
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument(
        "--optimizer",
        choices=["default", "lsgan", "wgan", "ranked"],
        default="ranked",
    )
    parser.add_argument("--ranker_list_size", type=int, default=128)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=256)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=3e-3)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/nondiff_binary_digits"),
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
    best_accuracy = objective.accuracy(best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"nondiff_binary_digits_seed_{args.seed}.npz",
        best_params=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_accuracy=np.asarray([best_accuracy], dtype=np.float32),
        sorted_values=np.asarray(
            optimizer.buffer.B.get_sorted_values(), dtype=np.float32
        ),
        n_params=np.asarray([best.numel()], dtype=np.int32),
        n_samples=np.asarray([args.n_samples], dtype=np.int32),
        task_arch=np.asarray([args.task_arch]),
        task_hidden_dim=np.asarray([args.task_hidden_dim], dtype=np.int32),
        circulant_width=np.asarray([args.circulant_width], dtype=np.int32),
        circulant_depth=np.asarray([args.circulant_depth], dtype=np.int32),
        conv_channels=np.asarray([args.conv_channels], dtype=np.int32),
        conv_depth=np.asarray([args.conv_depth], dtype=np.int32),
        residual_scale=np.asarray([args.residual_scale], dtype=np.float32),
        weight_mode=np.asarray([args.weight_mode]),
        noise=np.asarray([args.noise], dtype=np.float32),
        dropout=np.asarray([args.dropout], dtype=np.float32),
    )
    print(f"n_params={best.numel()}")
    print(f"best_value={best_value:.6f}")
    print(f"best_accuracy={best_accuracy:.4f}")
    print(f"saved={args.output_dir / f'nondiff_binary_digits_seed_{args.seed}.npz'}")


if __name__ == "__main__":
    main()
