import argparse
import json
import math
import random
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels, Rung
from gfog.curiosity import (
    CosineRamp,
    WarmupCosine,
    WarmupCosineAnnealing,
    WangIsolaUniformity,
    WangIsolaUniformityConfig,
)
from gfog.curiosity.scheduler import Scheduler
from gfog.models import MLP
from gfog.opt import (
    BaseOpt,
    DefaultOpt,
    HingeGANOpt,
    LSGANOpt,
    WGANOpt,
    WGANGPOpt,
    components,
)
from gfog.opt.latents_sampler import LatentSamplerBase, LatentSamplerLambda
from gfog.utils import uniformity_loss


EncodingMode = Literal[
    "direct",
    "coarse",
    "binary_coarse",
    "topk_volume",
    "sorted_material",
    "coarse_topk_volume",
    "coarse_residual",
    "soft_volume",
    "bar_primitives",
    "tiny_decoder",
]


LadderKind = Literal["volume", "compliance", "roughness", "connectivity", "diversity"]
LOAD_CASE_CHOICES = (
    "center_point",
    "right_top_point",
    "right_bottom_point",
    "right_two_points",
    "right_edge_uniform",
    "right_edge_shear",
    "tom_two_patches",
)
LoadCase = Literal[
    "center_point",
    "right_top_point",
    "right_bottom_point",
    "right_two_points",
    "right_edge_uniform",
    "right_edge_shear",
    "tom_two_patches",
]
RobustLoadAggregate = Literal["max", "mean", "cvar"]
ProblemPreset = Literal["default", "tom_cantilever_2d"]
ComplianceSolver = Literal["direct", "matrix_free_cg"]
MatrixFreeCGDType = Literal["float32", "float64"]
SortedMaterialProfile = Literal["binary", "linear", "sigmoid"]
ConvActivation = Literal["leaky_relu", "gelu", "silu"]
LatentDistribution = Literal["normal", "uniform"]
DesignProxyFn = Callable[[torch.Tensor], torch.Tensor]


def matrix_free_cg_torch_dtype(dtype_name: MatrixFreeCGDType) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float64":
        return torch.float64
    raise ValueError(f"Unknown matrix-free CG dtype: {dtype_name}")


class ConvDecoderGenerator(nn.Module):
    """Latent-to-grid convolutional decoder for topology score fields."""

    def __init__(
        self,
        latent_dim: int,
        output_height: int,
        output_width: int,
        channels: int = 64,
    ) -> None:
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width
        self.seed_height = max(2, math.ceil(output_height / 4))
        self.seed_width = max(2, math.ceil(output_width / 4))
        self.proj = nn.Linear(latent_dim, channels * self.seed_height * self.seed_width)
        self.net = nn.Sequential(
            nn.GroupNorm(8 if channels >= 8 else 1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).reshape(z.shape[0], -1, self.seed_height, self.seed_width)
        x = self.net(x)
        x = F.interpolate(
            x,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        )
        return x[:, 0].reshape(z.shape[0], -1)


class SetTransformerConvGenerator(nn.Module):
    """Batch-aware latent set transformer followed by a convolutional decoder."""

    def __init__(
        self,
        latent_dim: int,
        output_height: int,
        output_width: int,
        channels: int = 64,
        model_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"set generator model_dim must be divisible by heads, got {model_dim} and {heads}"
            )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, model_dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(model_dim)
        self._elite_context: torch.Tensor | None = None
        self.last_genomes: torch.Tensor | None = None
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.decoder = ConvDecoderGenerator(
            latent_dim=model_dim,
            output_height=output_height,
            output_width=output_width,
            channels=channels,
        )

    def set_elite_context(self, elite_context: torch.Tensor | None) -> None:
        self._elite_context = elite_context

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(
                f"SetTransformerConvGenerator expects 2D input, got {z.shape}"
            )
        latent_tokens = self.token_proj(z)
        if self._elite_context is None:
            tokens = latent_tokens.unsqueeze(0)
            genomes = self.encoder(tokens).squeeze(0)
        else:
            elite = self._elite_context.to(device=z.device, dtype=z.dtype)
            if elite.ndim != 2 or elite.shape[1] != latent_tokens.shape[1]:
                raise ValueError(
                    "elite genome context must have shape "
                    f"(n, {latent_tokens.shape[1]}), got {tuple(elite.shape)}"
                )
            query = latent_tokens.unsqueeze(0)
            memory = elite.unsqueeze(0)
            attended, _ = self.cross_attn(query=query, key=memory, value=memory)
            tokens = self.cross_norm(query + attended)
            genomes = self.encoder(tokens).squeeze(0)
        self.last_genomes = genomes
        return self.decoder(genomes)


class SetTransformerDirectGenerator(nn.Module):
    """Batch-aware generator that emits genome/design score vectors directly."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        model_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(
                f"set direct generator model_dim must be divisible by heads, got {model_dim} and {heads}"
            )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, model_dim),
            nn.GELU(),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, model_dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(model_dim)
        self._elite_context: torch.Tensor | None = None
        self.last_genomes: torch.Tensor | None = None
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

    def set_elite_context(self, elite_context: torch.Tensor | None) -> None:
        self._elite_context = elite_context

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(
                f"SetTransformerDirectGenerator expects 2D input, got {z.shape}"
            )
        latent_tokens = self.token_proj(z)
        if self._elite_context is None:
            tokens = latent_tokens.unsqueeze(0)
        else:
            elite = self._elite_context.to(device=z.device, dtype=z.dtype)
            context_tokens = self.context_proj(elite)
            query = latent_tokens.unsqueeze(0)
            memory = context_tokens.unsqueeze(0)
            attended, _ = self.cross_attn(query=query, key=memory, value=memory)
            tokens = self.cross_norm(query + attended)
        encoded = self.encoder(tokens).squeeze(0)
        scores = self.output_head(encoded)
        self.last_genomes = scores
        return scores


class NormalizedGenerator(nn.Module):
    """Wrap a generator and normalize each emitted genome/score vector."""

    def __init__(self, generator: nn.Module, mode: str, eps: float = 1e-8) -> None:
        super().__init__()
        if mode not in {"l2", "centered_l2", "layernorm"}:
            raise ValueError(f"Unknown generator normalization mode: {mode}")
        self.generator = generator
        self.mode = mode
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.generator(z)
        if self.mode == "l2":
            return F.normalize(x, p=2, dim=1, eps=self.eps)
        if self.mode == "centered_l2":
            centered = x - x.mean(dim=1, keepdim=True)
            return F.normalize(centered, p=2, dim=1, eps=self.eps)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.generator, name)


class FixedLatentBankSampler(LatentSamplerBase):
    """Sample optimizer batches from a fixed latent bank."""

    def __init__(
        self,
        bank: torch.Tensor,
        batch_size: int,
        mode: Literal["random", "shuffle_cycle", "balanced_niches"] = "shuffle_cycle",
        cluster_labels: torch.Tensor | None = None,
        noise_std: float = 0.0,
        normalize_noise_scale: bool = True,
    ) -> None:
        if bank.ndim != 2:
            raise ValueError(f"fixed latent bank must be 2D, got {tuple(bank.shape)}")
        if len(bank) < batch_size:
            raise ValueError(
                f"fixed latent bank size must be >= batch_size, got {len(bank)} and {batch_size}"
            )
        if mode not in {"random", "shuffle_cycle", "balanced_niches"}:
            raise ValueError(f"Unknown fixed latent sample mode: {mode}")
        if noise_std < 0:
            raise ValueError(
                f"fixed latent noise std must be non-negative, got {noise_std}"
            )
        self.bank = bank.detach().clone()
        self.batch_size = batch_size
        self.mode = mode
        self.noise_std = noise_std
        self.normalize_noise_scale = normalize_noise_scale
        self._order = torch.empty(0, dtype=torch.long, device=self.bank.device)
        self._position = 0
        self.cluster_labels: torch.Tensor | None = None
        self._cluster_member_indices: list[torch.Tensor] = []
        self._cluster_orders: list[torch.Tensor] = []
        self._cluster_positions: list[int] = []
        self._cluster_batch_counts: list[int] = []
        if mode == "balanced_niches":
            if cluster_labels is None:
                raise ValueError(
                    "fixed_latent_sample_mode=balanced_niches requires clustered latent labels"
                )
            labels = (
                cluster_labels.detach()
                .clone()
                .to(device=self.bank.device, dtype=torch.long)
            )
            if labels.ndim != 1 or len(labels) != len(self.bank):
                raise ValueError(
                    "fixed latent cluster_labels must be 1D with one label per bank entry"
                )
            unique_labels = torch.unique(labels, sorted=True)
            if len(unique_labels) < 2:
                raise ValueError(
                    "fixed_latent_sample_mode=balanced_niches requires at least two latent niches"
                )
            self.cluster_labels = labels
            self._cluster_member_indices = [
                torch.nonzero(labels == label, as_tuple=False).flatten()
                for label in unique_labels
            ]
            if min(len(indices) for indices in self._cluster_member_indices) <= 0:
                raise ValueError(
                    "fixed latent niches must all contain at least one entry"
                )
            self._cluster_orders = [
                torch.empty(0, dtype=torch.long, device=self.bank.device)
                for _ in self._cluster_member_indices
            ]
            self._cluster_positions = [0 for _ in self._cluster_member_indices]
            base = batch_size // len(self._cluster_member_indices)
            remainder = batch_size % len(self._cluster_member_indices)
            self._cluster_batch_counts = [
                base + (1 if cluster_idx < remainder else 0)
                for cluster_idx in range(len(self._cluster_member_indices))
            ]

    def __call__(self) -> torch.Tensor:
        if self.mode == "random":
            index = torch.randint(
                len(self.bank),
                (self.batch_size,),
                device=self.bank.device,
            )
            z = self.bank[index]
            return self._jitter(z)

        if self.mode == "balanced_niches":
            index = torch.cat(
                [
                    self._take_from_cluster(cluster_idx, count)
                    for cluster_idx, count in enumerate(self._cluster_batch_counts)
                    if count > 0
                ]
            )
            index = index[torch.randperm(len(index), device=self.bank.device)]
            z = self.bank[index]
            return self._jitter(z)

        if self._position + self.batch_size > len(self._order):
            self._order = torch.randperm(len(self.bank), device=self.bank.device)
            self._position = 0
        index = self._order[self._position : self._position + self.batch_size]
        self._position += self.batch_size
        z = self.bank[index]
        return self._jitter(z)

    def _take_from_cluster(self, cluster_idx: int, count: int) -> torch.Tensor:
        members = self._cluster_member_indices[cluster_idx]
        chunks: list[torch.Tensor] = []
        remaining = count
        while remaining > 0:
            if self._cluster_positions[cluster_idx] >= len(
                self._cluster_orders[cluster_idx]
            ):
                self._cluster_orders[cluster_idx] = members[
                    torch.randperm(len(members), device=self.bank.device)
                ]
                self._cluster_positions[cluster_idx] = 0
            position = self._cluster_positions[cluster_idx]
            order = self._cluster_orders[cluster_idx]
            take = min(remaining, len(order) - position)
            chunks.append(order[position : position + take])
            self._cluster_positions[cluster_idx] += take
            remaining -= take
        return torch.cat(chunks)

    def _jitter(self, z: torch.Tensor) -> torch.Tensor:
        if self.noise_std == 0:
            return z
        z = z + self.noise_std * torch.randn_like(z)
        if self.normalize_noise_scale:
            z = z / math.sqrt(1.0 + self.noise_std * self.noise_std)
        return z


class InitialBufferReplayGenerator(nn.Module):
    """Replay fixed genome codes during initial buffer fill, then defer to G."""

    def __init__(self, codes: torch.Tensor, fallback: nn.Module) -> None:
        super().__init__()
        if codes.ndim != 2:
            raise ValueError(
                f"initial buffer codes must be 2D, got {tuple(codes.shape)}"
            )
        self.register_buffer("codes", codes.detach().clone())
        self.fallback = fallback
        self.position = 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        if self.position + batch_size <= self.codes.shape[0]:
            out = self.codes[self.position : self.position + batch_size]
            self.position += batch_size
            return out.to(device=z.device, dtype=z.dtype)
        return self.fallback(z)


def sample_latents(
    count: int,
    latent_dim: int,
    *,
    distribution: LatentDistribution,
    uniform_low: float,
    uniform_high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample latent vectors from the configured prior."""
    if count <= 0:
        raise ValueError(f"latent sample count must be positive, got {count}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if distribution == "normal":
        return torch.randn(count, latent_dim, device=device, dtype=dtype)
    if distribution == "uniform":
        if uniform_low >= uniform_high:
            raise ValueError(
                f"latent_uniform_low must be < latent_uniform_high, got {uniform_low} and {uniform_high}"
            )
        return torch.empty(count, latent_dim, device=device, dtype=dtype).uniform_(
            uniform_low,
            uniform_high,
        )
    raise ValueError(f"Unknown latent distribution: {distribution}")


class CombinedCuriosityLoss(nn.Module):
    """Add multiple G-side curiosity/diversity losses."""

    def __init__(self, losses: list[nn.Module]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=g_out.device, dtype=g_out.dtype)
        for module in self.losses:
            loss = loss + module(g_out)
        return loss


class FixedLatentBankUniformity(nn.Module):
    """Uniformity over current G outputs for a fixed latent bank."""

    def __init__(
        self,
        generator: nn.Module,
        bank: torch.Tensor,
        *,
        weight: float,
        batch_size: int,
        t: float = 2.0,
        sample_mode: Literal["random", "shuffle_cycle"] = "shuffle_cycle",
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(
                f"fixed_latent_uniformity_weight must be non-negative, got {weight}"
            )
        if bank.ndim != 2:
            raise ValueError(f"fixed latent bank must be 2D, got {tuple(bank.shape)}")
        if batch_size < 2:
            raise ValueError(
                f"fixed_latent_uniformity_batch_size must be >= 2, got {batch_size}"
            )
        if sample_mode not in {"random", "shuffle_cycle"}:
            raise ValueError(
                f"Unknown fixed latent uniformity sample mode: {sample_mode}"
            )
        self.generator = generator
        self.register_buffer("bank", bank.detach().clone())
        self.weight = weight
        self.batch_size = min(batch_size, len(bank))
        self.t = t
        self.sample_mode = sample_mode
        self.register_buffer(
            "_order", torch.empty(0, dtype=torch.long), persistent=False
        )
        self._position = 0

    def _sample_bank(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bank = self.bank.to(device=device, dtype=dtype)
        if self.sample_mode == "random":
            index = torch.randint(len(bank), (self.batch_size,), device=device)
            return bank[index]
        if self._position + self.batch_size > len(self._order):
            self._order = torch.randperm(len(bank), device=device)
            self._position = 0
        index = self._order[self._position : self._position + self.batch_size]
        self._position += self.batch_size
        return bank[index]

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        z = self._sample_bank(g_out.device, g_out.dtype)
        bank_outputs = self.generator(z).reshape(z.shape[0], -1)
        return self.weight * uniformity_loss(bank_outputs, t=self.t)


class NicheOutputSeparationLoss(nn.Module):
    """Repel generated outputs assigned to different stable niche anchors."""

    def __init__(
        self,
        *,
        buffer: Any,
        design_proxy: DesignProxyFn,
        weight: float,
        margin: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(
                f"niche_output_separation_weight must be non-negative, got {weight}"
            )
        if margin <= 0:
            raise ValueError(
                f"niche_output_separation_margin must be positive, got {margin}"
            )
        if not hasattr(buffer, "niche_anchor_proxies"):
            raise ValueError("niche output separation requires a NicheEliteBuffer")
        self.buffer = buffer
        self.design_proxy = design_proxy
        self.weight = weight
        self.margin = margin
        self.eps = eps

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((), device=g_out.device, dtype=g_out.dtype)
        if self.weight <= 0 or g_out.shape[0] < 2:
            return zero

        anchors = [
            anchor.reshape(-1).detach().to(torch.bool).cpu()
            for anchor in self.buffer.niche_anchor_proxies
            if anchor is not None
        ]
        if len(anchors) < 2:
            return zero

        with torch.no_grad():
            proxies = self.design_proxy(g_out.detach())
            if proxies.shape[0] != g_out.shape[0]:
                raise ValueError(
                    "niche output separation design_proxy must return one proxy "
                    f"per generated output, got {proxies.shape[0]} for {g_out.shape[0]}"
                )
            proxy_flat = proxies.reshape(proxies.shape[0], -1).to(torch.bool).cpu()
            anchor_flat = torch.stack(anchors)
            if proxy_flat.shape[1] != anchor_flat.shape[1]:
                raise ValueError(
                    "niche output separation proxy dimension mismatch: "
                    f"{proxy_flat.shape[1]} vs {anchor_flat.shape[1]}"
                )
            distances = (proxy_flat[:, None, :] != anchor_flat[None, :, :]).to(
                torch.float32
            )
            labels = distances.mean(dim=2).argmin(dim=1).to(g_out.device)

        flat = g_out.reshape(g_out.shape[0], -1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        flat = F.normalize(flat, p=2, dim=1, eps=self.eps)

        group_means: list[torch.Tensor] = []
        for label in torch.unique(labels, sorted=True):
            group = flat[labels == label]
            if group.shape[0] == 0:
                continue
            mean = group.mean(dim=0)
            group_means.append(F.normalize(mean, p=2, dim=0, eps=self.eps))
        if len(group_means) < 2:
            return zero

        pairwise_distances = torch.pdist(torch.stack(group_means), p=2)
        if pairwise_distances.numel() == 0:
            return zero
        penalty = F.relu(self.margin - pairwise_distances).pow(2).mean()
        return self.weight * penalty


class RankValueMLP(nn.Module):
    """MLP with an original rank head plus an auxiliary value head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: nn.Module | None = None,
        use_spectral_norm: bool = False,
        disable_bias: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims}")
        self.activation = activation if activation is not None else nn.GELU()
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda layer: layer)
        dims = [input_dim] + list(hidden_dims)
        self.hidden_layers = nn.ModuleList(
            [
                sn(nn.Linear(dims[i], dims[i + 1], bias=(not disable_bias)))
                for i in range(len(dims) - 1)
            ]
        )
        head_input_dim = dims[-1]
        # Keep this before value_head so the rank path matches the original MLP
        # initialization order as closely as possible.
        self.rank_head = sn(nn.Linear(head_input_dim, 1, bias=(not disable_bias)))
        self.value_head = sn(nn.Linear(head_input_dim, 1, bias=(not disable_bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        rank = self.rank_head(x)
        value = self.value_head(x)
        return torch.cat([rank, value], dim=-1)


@torch.no_grad()
def select_output_diverse_latent_bank(
    generator: nn.Module,
    *,
    latent_dim: int,
    bank_size: int,
    candidate_multiplier: int,
    chunk_size: int,
    distribution: LatentDistribution,
    uniform_low: float,
    uniform_high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Greedily select fixed latents whose initial generator outputs differ."""
    if bank_size <= 0:
        raise ValueError(f"fixed_latent_bank_size must be > 0, got {bank_size}")
    if candidate_multiplier < 1:
        raise ValueError(
            f"fixed_latent_candidate_multiplier must be >= 1, got {candidate_multiplier}"
        )
    if chunk_size <= 0:
        raise ValueError(f"fixed_latent_chunk_size must be > 0, got {chunk_size}")

    candidate_count = bank_size * candidate_multiplier
    candidates = sample_latents(
        candidate_count,
        latent_dim,
        distribution=distribution,
        uniform_low=uniform_low,
        uniform_high=uniform_high,
        device=device,
        dtype=dtype,
    )
    outputs = []
    was_training = generator.training
    generator.eval()
    for start in range(0, candidate_count, chunk_size):
        chunk = candidates[start : start + chunk_size]
        out = generator(chunk).reshape(len(chunk), -1)
        out = out - out.mean(dim=1, keepdim=True)
        outputs.append(F.normalize(out, p=2, dim=1, eps=1e-8).cpu())
    if was_training:
        generator.train()

    output_bank = torch.cat(outputs, dim=0)
    selected = torch.empty(bank_size, dtype=torch.long)
    min_distance = torch.full((candidate_count,), float("inf"))
    current = torch.randint(candidate_count, (1,)).item()
    for i in range(bank_size):
        selected[i] = current
        distance = (output_bank - output_bank[current]).pow(2).sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance[selected[: i + 1]] = -1.0
        current = int(torch.argmax(min_distance).item())

    return candidates[selected].detach()


@torch.no_grad()
def make_clustered_niche_latent_bank(
    *,
    latent_dim: int,
    bank_size: int,
    niche_count: int,
    center_scale: float,
    within_std: float,
    normalize_radius: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create fixed latent niches with larger between-niche than within-niche gaps."""
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if bank_size <= 0:
        raise ValueError(f"fixed_latent_bank_size must be > 0, got {bank_size}")
    if niche_count <= 1:
        raise ValueError(f"fixed_latent_niche_count must be >= 2, got {niche_count}")
    if niche_count > bank_size:
        raise ValueError(
            f"fixed_latent_niche_count must not exceed bank size, got {niche_count} and {bank_size}"
        )
    if center_scale <= 0:
        raise ValueError(
            f"fixed_latent_niche_center_scale must be positive, got {center_scale}"
        )
    if within_std <= 0:
        raise ValueError(
            f"fixed_latent_niche_within_std must be positive, got {within_std}"
        )

    if niche_count == 2:
        center = F.normalize(
            torch.randn(1, latent_dim, device=device, dtype=dtype),
            p=2,
            dim=1,
            eps=1e-8,
        )
        centers = torch.cat([center, -center], dim=0)
    else:
        candidate_count = max(256, niche_count * 128)
        candidates = F.normalize(
            torch.randn(candidate_count, latent_dim, device=device, dtype=dtype),
            p=2,
            dim=1,
            eps=1e-8,
        )
        selected = torch.empty(niche_count, dtype=torch.long, device=device)
        selected[0] = int(torch.randint(candidate_count, (1,), device=device).item())
        min_distance = torch.full(
            (candidate_count,), float("inf"), device=device, dtype=dtype
        )
        for idx in range(1, niche_count):
            previous = int(selected[idx - 1].item())
            distance = torch.cdist(candidates, candidates[previous : previous + 1])
            distance = distance.flatten()
            min_distance = torch.minimum(min_distance, distance)
            min_distance[selected[:idx]] = -1.0
            selected[idx] = int(torch.argmax(min_distance).item())
        centers = candidates[selected]

    capacities = niche_capacities(bank_size, niche_count)
    bank_parts = []
    label_parts = []
    for niche_idx, capacity in enumerate(capacities):
        centered = center_scale * centers[niche_idx].unsqueeze(0)
        noise = within_std * torch.randn(
            capacity, latent_dim, device=device, dtype=dtype
        )
        bank_parts.append(centered + noise)
        label_parts.append(
            torch.full((capacity,), niche_idx, device=device, dtype=torch.long)
        )
    bank = torch.cat(bank_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    if normalize_radius:
        bank = math.sqrt(latent_dim) * F.normalize(bank, p=2, dim=1, eps=1e-8)

    permutation = torch.randperm(len(bank), device=device)
    bank = bank[permutation].detach()
    labels = labels[permutation].detach()

    distances = torch.cdist(bank, bank)
    same_label = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(len(bank), dtype=torch.bool, device=device)
    within = distances[same_label & off_diagonal]
    between = distances[~same_label]
    center_distances = torch.cdist(centers, centers)
    center_between = center_distances[
        ~torch.eye(niche_count, dtype=torch.bool, device=device)
    ]

    def finite_stat(values: torch.Tensor, reducer: str) -> float:
        if len(values) == 0:
            return float("nan")
        if reducer == "min":
            return float(values.min().item())
        if reducer == "max":
            return float(values.max().item())
        return float(values.mean().item())

    stats = {
        "latent_niche_within_mean_l2": finite_stat(within, "mean"),
        "latent_niche_within_max_l2": finite_stat(within, "max"),
        "latent_niche_between_mean_l2": finite_stat(between, "mean"),
        "latent_niche_between_min_l2": finite_stat(between, "min"),
        "latent_niche_center_mean_l2": finite_stat(center_between, "mean"),
    }
    stats["latent_niche_separation_margin_l2"] = (
        stats["latent_niche_between_min_l2"] - stats["latent_niche_within_max_l2"]
    )
    return bank, labels, stats


class ConvDiscriminator(nn.Module):
    """Small CNN discriminator for flattened grids."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: int = 32,
        use_spectral_norm: bool = True,
        activation: ConvActivation = "leaky_relu",
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.input_height = input_height
        self.input_width = input_width
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda layer: layer)

        def act() -> nn.Module:
            if activation == "leaky_relu":
                return nn.LeakyReLU(0.2)
            if activation == "gelu":
                return nn.GELU()
            if activation == "silu":
                return nn.SiLU()
            raise ValueError(f"Unknown conv discriminator activation: {activation}")

        self.net = nn.Sequential(
            sn(nn.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)),
            act(),
            sn(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)),
            act(),
            sn(
                nn.Conv2d(
                    channels * 2, channels * 4, kernel_size=3, stride=2, padding=1
                )
            ),
            act(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(channels * 4, output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, self.input_height, self.input_width)
        return self.net(x)


class SetTransformerDiscriminator(nn.Module):
    """Permutation-equivariant listwise discriminator over a candidate batch."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if model_dim % heads != 0:
            raise ValueError(
                f"set transformer model_dim must be divisible by heads, got {model_dim} and {heads}"
            )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.score_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"SetTransformerDiscriminator expects 2D input, got {x.shape}"
            )
        tokens = self.input_proj(x).unsqueeze(0)
        encoded = self.encoder(tokens).squeeze(0)
        return self.score_head(encoded)


@dataclass
class FEMConfig:
    grid_width: int = 32
    grid_height: int = 16
    domain_width: float = 1.0
    domain_height: float = 1.0
    coarse_grid_width: int | None = None
    coarse_grid_height: int | None = None
    encoding: EncodingMode = "direct"
    residual_scale: float = 0.25
    simp_p: float = 3.0
    e_min: float = 1e-3
    e_max: float = 1.0
    poisson_ratio: float = 0.3
    volume_max: float = 0.48
    roughness_max: float = 0.18
    connectivity_max: float | None = None
    volume_ladder: tuple[float, ...] = ()
    compliance_ladder: tuple[float, ...] = ()
    roughness_ladder: tuple[float, ...] = ()
    connectivity_ladder: tuple[float, ...] = ()
    diversity_ladder: tuple[float, ...] = ()
    ladder_sequence: tuple[tuple[LadderKind, float], ...] = ()
    use_levels_ladder: bool = False
    levels_ladder_objectives: tuple[str, ...] = ()
    diversity_reference_size: int = 32
    diversity_chamfer_max_points: int = 256
    load_scale: float = 1.0
    load_case: LoadCase = "center_point"
    robust_load_cases: tuple[LoadCase, ...] = ()
    robust_load_aggregate: RobustLoadAggregate = "max"
    robust_load_cvar_frac: float = 0.5
    removal_ladder_volumes: tuple[float, ...] = ()
    removal_ladder_compliances: tuple[float, ...] = ()
    removal_ladder_connectivity_max: float | None = None
    fem_workers: int = 1
    compliance_solver: ComplianceSolver = "direct"
    matrix_free_cg_max_iter: int = 1000
    matrix_free_cg_tol: float = 1e-6
    matrix_free_cg_device: str = "cpu"
    matrix_free_cg_dtype: MatrixFreeCGDType = "float64"
    sorted_material_profile: SortedMaterialProfile = "linear"
    sorted_material_steepness: float = 12.0
    density_filter_radius: int = 1
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    hard_binarize: bool = False
    binhead_connect_support: bool = False
    tiny_decoder_model: str = "madebyollin/taesd"
    tiny_decoder_latent_channels: int = 4
    tiny_decoder_latent_height: int = 8
    tiny_decoder_latent_width: int = 8
    tiny_decoder_latent_scale: float = 1.0
    bar_count: int = 16
    bar_width_min: float = 0.02
    bar_width_max: float = 0.08
    bar_edge_softness: float = 0.01


def decode_design_logits_numpy(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


def decode_design_logits_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def sigmoid_with_target_mean_numpy(
    logits: np.ndarray,
    target_mean: float,
    *,
    n_steps: int = 40,
) -> np.ndarray:
    """Shift logits so sigmoid densities have the requested per-sample mean."""
    x = np.asarray(logits, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(axis=1, keepdims=True) - 40.0
    hi = flat.max(axis=1, keepdims=True) + 40.0
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        density = 1.0 / (1.0 + np.exp(-(flat - mid)))
        too_dense = density.mean(axis=1, keepdims=True) > target_mean
        lo = np.where(too_dense, mid, lo)
        hi = np.where(too_dense, hi, mid)
    tau = 0.5 * (lo + hi)
    return (1.0 / (1.0 + np.exp(-(flat - tau)))).reshape(x.shape)


def sigmoid_with_target_mean_torch(
    logits: torch.Tensor,
    target_mean: float,
    *,
    n_steps: int = 40,
) -> torch.Tensor:
    """Differentiably shift logits so sigmoid densities hit target mean."""
    flat = logits.reshape(logits.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values - 40.0
    hi = flat.max(dim=1, keepdim=True).values + 40.0
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        density = torch.sigmoid(flat - mid)
        too_dense = density.mean(dim=1, keepdim=True) > target_mean
        lo = torch.where(too_dense, mid, lo)
        hi = torch.where(too_dense, hi, mid)
    tau = 0.5 * (lo + hi)
    return torch.sigmoid(flat - tau).reshape_as(logits)


def expand_design_code_numpy(
    x: np.ndarray,
    *,
    coarse_height: int,
    coarse_width: int,
    full_height: int,
    full_width: int,
) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32).reshape(-1, 1, coarse_height, coarse_width)
    x_torch = torch.from_numpy(x32)
    with torch.no_grad():
        up = F.interpolate(
            x_torch,
            size=(full_height, full_width),
            mode="bilinear",
            align_corners=False,
        )
    return up[:, 0].cpu().numpy()


def project_positive_vector_to_binary_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError("binhead projection expects non-negative entries")
    norm = np.linalg.norm(x)
    if norm <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    v = x / norm
    p = np.argsort(-v)
    sorted_v = v[p]
    scalers = 1.0 / np.sqrt(np.arange(1, v.size + 1, dtype=np.float64))
    scores = scalers * np.cumsum(sorted_v)
    idx = int(np.argmax(scores))
    out = np.zeros_like(v, dtype=np.float64)
    out[p[: idx + 1]] = 1.0
    return out


def project_scores_to_topk_binary_numpy(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(scores, dtype=np.float64)
    if k <= 0:
        return out
    k = min(k, scores.size)
    top_idx = np.argpartition(scores, -k)[-k:]
    out[top_idx] = 1.0
    return out


def fixed_sorted_material_values_numpy(
    n: int,
    target_mean: float,
    *,
    profile: SortedMaterialProfile,
    steepness: float = 12.0,
) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= target_mean <= 1.0:
        raise ValueError(f"target_mean must be in [0, 1], got {target_mean}")
    if profile == "binary":
        k = int(round(target_mean * n))
        values = np.zeros(n, dtype=np.float64)
        values[:k] = 1.0
        return values
    if profile == "linear":
        values = np.linspace(1.0, 0.0, n, dtype=np.float64)
        mean = float(values.mean())
        return np.clip(values * target_mean / max(mean, 1e-12), 0.0, 1.0)
    if profile == "sigmoid":
        if steepness <= 0:
            raise ValueError(f"steepness must be positive, got {steepness}")
        ranks = (np.arange(n, dtype=np.float64) + 0.5) / n
        logits = -steepness * (ranks - target_mean)
        values = 1.0 / (1.0 + np.exp(-logits))
        lo = values.min()
        hi = values.max()
        if hi - lo > 1e-12:
            values = (values - lo) / (hi - lo)
        mean = float(values.mean())
        return np.clip(values * target_mean / max(mean, 1e-12), 0.0, 1.0)
    raise ValueError(f"Unknown sorted material profile: {profile}")


def project_scores_to_fixed_sorted_material_numpy(
    scores: np.ndarray,
    material_values: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = np.asarray(material_values, dtype=np.float64).reshape(-1)
    if scores.size != values.size:
        raise ValueError(
            f"scores and material_values must have same size, got {scores.size} and {values.size}"
        )
    order = np.argsort(-scores, kind="stable")
    out = np.empty_like(scores, dtype=np.float64)
    out[order] = values
    return out


def random_blob_designs_numpy(
    count: int,
    *,
    nely: int,
    nelx: int,
    target_mean: float,
    rng: np.random.Generator,
    blob_count_min: int,
    blob_count_max: int,
    radius_min: float,
    radius_max: float,
) -> np.ndarray:
    """Create binary designs by placing random smooth material blobs."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if blob_count_min < 1:
        raise ValueError(f"blob_count_min must be >= 1, got {blob_count_min}")
    if blob_count_max < blob_count_min:
        raise ValueError(
            "blob_count_max must be >= blob_count_min, got "
            f"{blob_count_max} and {blob_count_min}"
        )
    if not 0.0 < radius_min <= radius_max:
        raise ValueError(
            f"blob radii must satisfy 0 < min <= max, got {radius_min}, {radius_max}"
        )
    n = nely * nelx
    k = min(max(int(round(target_mean * n)), 1), n)
    yy = (np.arange(nely, dtype=np.float64) + 0.5) / float(nely)
    xx = (np.arange(nelx, dtype=np.float64) + 0.5) / float(nelx)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    designs = np.zeros((count, nely, nelx), dtype=np.float32)
    for idx in range(count):
        field = np.zeros((nely, nelx), dtype=np.float64)
        n_blobs = int(rng.integers(blob_count_min, blob_count_max + 1))
        for _ in range(n_blobs):
            cx = float(rng.uniform(0.0, 1.0))
            cy = float(rng.uniform(0.0, 1.0))
            rx = float(rng.uniform(radius_min, radius_max))
            ry = float(rng.uniform(radius_min, radius_max))
            strength = float(rng.uniform(0.5, 1.5))
            dist = ((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2
            field += strength * np.exp(-0.5 * dist)
        field += 1e-3 * rng.standard_normal(size=field.shape)
        flat = field.reshape(-1)
        top_idx = np.argpartition(flat, -k)[-k:]
        design = np.zeros(n, dtype=np.float32)
        design[top_idx] = 1.0
        designs[idx] = design.reshape(nely, nelx)
    return designs


def mean_pairwise_hamming_numpy(designs: np.ndarray) -> float:
    flat = np.asarray(designs >= 0.5).reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    dists = flat[:, None, :] != flat[None, :, :]
    triu = np.triu_indices(flat.shape[0], k=1)
    return float(dists.mean(axis=-1)[triu].mean())


def min_pairwise_hamming_numpy(designs: np.ndarray) -> float:
    flat = np.asarray(designs >= 0.5).reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    dists = flat[:, None, :] != flat[None, :, :]
    triu = np.triu_indices(flat.shape[0], k=1)
    return float(dists.mean(axis=-1)[triu].min())


def select_diverse_binary_designs_numpy(
    designs: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    min_hamming: float,
) -> np.ndarray:
    """Deduplicate and greedily select binary designs by max-min Hamming distance."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if min_hamming < 0:
        raise ValueError(
            f"initial_blob_min_hamming must be non-negative, got {min_hamming}"
        )

    flat = np.asarray(designs >= 0.5).reshape(designs.shape[0], -1)
    _unique_flat, unique_idx = np.unique(flat, axis=0, return_index=True)
    candidate_idx = np.sort(unique_idx)
    candidates = flat[candidate_idx]
    if candidates.shape[0] <= count:
        return designs[candidate_idx].astype(np.float32, copy=False)

    selected: list[int] = [int(rng.integers(0, candidates.shape[0]))]
    remaining = np.ones(candidates.shape[0], dtype=bool)
    remaining[selected[0]] = False
    min_dist = (candidates != candidates[selected[0]]).mean(axis=1)
    while len(selected) < count:
        feasible = remaining & (min_dist >= min_hamming)
        pool = feasible if np.any(feasible) else remaining
        next_idx = int(np.argmax(np.where(pool, min_dist, -1.0)))
        selected.append(next_idx)
        remaining[next_idx] = False
        new_dist = (candidates != candidates[next_idx]).mean(axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return candidates[selected].reshape(count, *designs.shape[1:]).astype(np.float32)


def niche_capacities(count: int, niche_count: int) -> list[int]:
    """Split a count as evenly as possible across niches."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if niche_count <= 0:
        raise ValueError(f"niche_count must be positive, got {niche_count}")
    base = count // niche_count
    remainder = count % niche_count
    capacities = [base + (1 if idx < remainder else 0) for idx in range(niche_count)]
    if min(capacities) <= 0:
        raise ValueError(
            f"niche_count must not exceed count, got {niche_count} for count={count}"
        )
    return capacities


def binary_cluster_hamming_stats(
    designs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Measure within-cluster and between-cluster Hamming distances."""
    flat = np.asarray(designs >= 0.5).reshape(designs.shape[0], -1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if flat.shape[0] != labels.shape[0]:
        raise ValueError(
            f"labels length must match designs, got {labels.shape[0]} and {flat.shape[0]}"
        )
    if flat.shape[0] < 2:
        return {
            "cluster_internal_mean_hamming": 0.0,
            "cluster_external_mean_hamming": 0.0,
            "cluster_separation_margin": 0.0,
            "cluster_internal_min_hamming": 0.0,
            "cluster_external_min_hamming": 0.0,
        }
    distances = (flat[:, None, :] != flat[None, :, :]).mean(axis=-1)
    triu = np.triu_indices(flat.shape[0], k=1)
    same = labels[:, None] == labels[None, :]
    internal = distances[triu][same[triu]]
    external = distances[triu][~same[triu]]
    internal_mean = float(internal.mean()) if internal.size else 0.0
    external_mean = float(external.mean()) if external.size else 0.0
    internal_min = float(internal.min()) if internal.size else 0.0
    external_min = float(external.min()) if external.size else 0.0
    return {
        "cluster_internal_mean_hamming": internal_mean,
        "cluster_external_mean_hamming": external_mean,
        "cluster_separation_margin": external_mean - internal_mean,
        "cluster_internal_min_hamming": internal_min,
        "cluster_external_min_hamming": external_min,
    }


def select_clustered_binary_designs_numpy(
    designs: np.ndarray,
    *,
    count: int,
    niche_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Select binary designs grouped into balanced Hamming-distance niches."""
    capacities = niche_capacities(count, niche_count)
    flat = np.asarray(designs >= 0.5).reshape(designs.shape[0], -1)
    _unique_flat, unique_idx = np.unique(flat, axis=0, return_index=True)
    candidate_idx = np.sort(unique_idx)
    candidates = flat[candidate_idx]
    if candidates.shape[0] < count:
        raise ValueError(
            "random blob initializer produced fewer unique designs than required: "
            f"{candidates.shape[0]} < {count}. Increase --initial_blob_candidate_multiplier."
        )

    medoids: list[int] = [int(rng.integers(0, candidates.shape[0]))]
    min_dist = (candidates != candidates[medoids[0]]).mean(axis=1)
    for _ in range(1, niche_count):
        min_dist[medoids] = -1.0
        next_medoid = int(np.argmax(min_dist))
        medoids.append(next_medoid)
        new_dist = (candidates != candidates[next_medoid]).mean(axis=1)
        min_dist = np.minimum(min_dist, new_dist)

    distance_to_medoids = np.stack(
        [(candidates != candidates[medoid]).mean(axis=1) for medoid in medoids],
        axis=1,
    )
    selected_indices: list[int] = []
    selected_labels: list[int] = []
    used = np.zeros(candidates.shape[0], dtype=bool)
    for niche_idx, capacity in enumerate(capacities):
        order = np.argsort(distance_to_medoids[:, niche_idx])
        chosen = []
        for idx in order:
            idx = int(idx)
            if used[idx]:
                continue
            chosen.append(idx)
            used[idx] = True
            if len(chosen) >= capacity:
                break
        if len(chosen) < capacity:
            raise ValueError(
                f"Could not fill niche {niche_idx}: selected {len(chosen)} of {capacity}"
            )
        selected_indices.extend(chosen)
        selected_labels.extend([niche_idx] * len(chosen))

    selected = (
        candidates[selected_indices]
        .reshape(count, *designs.shape[1:])
        .astype(np.float32)
    )
    labels = np.asarray(selected_labels, dtype=np.int32)
    stats = binary_cluster_hamming_stats(selected, labels)
    medoid_distances = np.asarray(
        [
            (candidates[a] != candidates[b]).mean()
            for i, a in enumerate(medoids)
            for b in medoids[i + 1 :]
        ],
        dtype=np.float64,
    )
    stats["cluster_medoid_min_hamming"] = (
        float(medoid_distances.min()) if medoid_distances.size else 0.0
    )
    stats["cluster_medoid_mean_hamming"] = (
        float(medoid_distances.mean()) if medoid_distances.size else 0.0
    )
    return selected, labels, stats


def sorted_material_scores_from_binary_designs_numpy(
    designs: np.ndarray,
    *,
    rng: np.random.Generator,
    margin: float,
    noise: float,
) -> np.ndarray:
    """Encode binary masks as sorted-material score vectors."""
    if margin <= 0:
        raise ValueError(f"initial_blob_score_margin must be positive, got {margin}")
    if noise < 0:
        raise ValueError(f"initial_blob_score_noise must be non-negative, got {noise}")
    flat = np.asarray(designs, dtype=np.float32).reshape(designs.shape[0], -1)
    scores = np.where(flat >= 0.5, margin, -margin).astype(np.float32)
    if noise > 0:
        scores += noise * rng.standard_normal(size=scores.shape).astype(np.float32)
    return scores


def make_sorted_material_blob_seed_codes(
    count: int,
    *,
    nely: int,
    nelx: int,
    target_mean: float,
    seed: int,
    blob_count_min: int,
    blob_count_max: int,
    radius_min: float,
    radius_max: float,
    score_margin: float,
    score_noise: float,
    candidate_multiplier: int,
    min_hamming: float,
    cluster_niche_count: int = 1,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if candidate_multiplier < 1:
        raise ValueError(
            "initial_blob_candidate_multiplier must be >= 1, got "
            f"{candidate_multiplier}"
        )
    rng = np.random.default_rng(seed)
    candidate_count = max(count, count * candidate_multiplier)
    designs = random_blob_designs_numpy(
        candidate_count,
        nely=nely,
        nelx=nelx,
        target_mean=target_mean,
        rng=rng,
        blob_count_min=blob_count_min,
        blob_count_max=blob_count_max,
        radius_min=radius_min,
        radius_max=radius_max,
    )
    if cluster_niche_count > 1:
        selected, labels, cluster_stats = select_clustered_binary_designs_numpy(
            designs,
            count=count,
            niche_count=cluster_niche_count,
            rng=rng,
        )
    else:
        selected = select_diverse_binary_designs_numpy(
            designs,
            count=count,
            rng=rng,
            min_hamming=min_hamming,
        )
        labels = np.zeros(count, dtype=np.int32)
        cluster_stats = {}
    if selected.shape[0] < count:
        raise ValueError(
            "random blob initializer produced fewer unique designs than required: "
            f"{selected.shape[0]} < {count}. Increase --initial_blob_candidate_multiplier."
        )
    stats = {
        "candidate_count": float(candidate_count),
        "selected_count": float(selected.shape[0]),
        "mean_pairwise_hamming": mean_pairwise_hamming_numpy(selected),
        "min_pairwise_hamming": min_pairwise_hamming_numpy(selected),
        "cluster_niche_count": float(cluster_niche_count),
        **cluster_stats,
    }
    scores = sorted_material_scores_from_binary_designs_numpy(
        selected,
        rng=rng,
        margin=score_margin,
        noise=score_noise,
    )
    return scores, stats, labels


def rasterize_bar_primitives_numpy(
    params: np.ndarray,
    *,
    nely: int,
    nelx: int,
    width_min: float,
    width_max: float,
    edge_softness: float,
) -> np.ndarray:
    """Rasterize normalized line-segment primitives into density fields."""
    if width_min <= 0:
        raise ValueError(f"width_min must be positive, got {width_min}")
    if width_max < width_min:
        raise ValueError(
            f"width_max must be >= width_min, got {width_max} < {width_min}"
        )
    if edge_softness <= 0:
        raise ValueError(f"edge_softness must be positive, got {edge_softness}")

    raw = np.asarray(params, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 5:
        raise ValueError(
            f"bar params must have shape (batch, bars, 5), got {raw.shape}"
        )

    batch_size, bar_count, _ = raw.shape
    xs = (np.arange(nelx, dtype=np.float64) + 0.5) / float(nelx)
    ys = (np.arange(nely, dtype=np.float64) + 0.5) / float(nely)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x, grid_y], axis=-1).reshape(1, 1, nely * nelx, 2)

    coords = 1.0 / (1.0 + np.exp(-raw[..., :4]))
    starts = coords[..., :2]
    ends = coords[..., 2:4]
    widths = width_min + (width_max - width_min) / (1.0 + np.exp(-raw[..., 4]))

    segment = ends - starts
    denom = np.sum(segment * segment, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    rel = points - starts[:, :, None, :]
    t = (
        np.sum(rel * segment[:, :, None, :], axis=-1, keepdims=True)
        / denom[:, :, None, :]
    )
    t = np.clip(t, 0.0, 1.0)
    closest = starts[:, :, None, :] + t * segment[:, :, None, :]
    dist = np.linalg.norm(points - closest, axis=-1)
    logits = (widths[:, :, None] - dist) / edge_softness
    bar_density = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    density = np.max(bar_density, axis=1)
    return density.reshape(batch_size, nely, nelx)


def support_connected_mask_numpy(binary: np.ndarray) -> np.ndarray:
    solid = np.asarray(binary, dtype=bool)
    height, width = solid.shape
    visited = np.zeros_like(solid, dtype=bool)
    stack: list[tuple[int, int]] = [
        (row, 0) for row in range(height) if bool(solid[row, 0])
    ]
    for row, col in stack:
        visited[row, col] = True

    while stack:
        row, col = stack.pop()
        for nr, nc in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= nr < height
                and 0 <= nc < width
                and bool(solid[nr, nc])
                and not bool(visited[nr, nc])
            ):
                visited[nr, nc] = True
                stack.append((nr, nc))
    return visited


def connect_support_and_refill_numpy(
    binary: np.ndarray,
    scores: np.ndarray,
    *,
    target_count: int,
) -> np.ndarray:
    """Keep support-connected material, then refill nearby high-score cells."""
    solid = np.asarray(binary, dtype=np.float64) >= 0.5
    score_grid = np.asarray(scores, dtype=np.float64).reshape(solid.shape)
    target_count = min(max(target_count, 0), solid.size)
    if target_count == 0:
        return np.zeros_like(binary, dtype=np.float64)

    connected = support_connected_mask_numpy(solid)
    if not np.any(connected):
        flat_scores = score_grid.reshape(-1)
        start = int(np.argmax(flat_scores))
        connected.reshape(-1)[start] = True

    selected = connected.copy()
    if int(np.count_nonzero(selected)) > target_count:
        connected_scores = np.where(selected, score_grid, -np.inf).reshape(-1)
        keep_idx = np.argpartition(connected_scores, -target_count)[-target_count:]
        trimmed = np.zeros_like(selected)
        trimmed.reshape(-1)[keep_idx] = True
        selected = support_connected_mask_numpy(trimmed)

    while int(np.count_nonzero(selected)) < target_count:
        candidates = np.zeros_like(selected, dtype=bool)
        candidates[:-1, :] |= selected[1:, :]
        candidates[1:, :] |= selected[:-1, :]
        candidates[:, :-1] |= selected[:, 1:]
        candidates[:, 1:] |= selected[:, :-1]
        candidates &= ~selected
        if not np.any(candidates):
            candidates = ~selected
        candidate_scores = np.where(candidates, score_grid, -np.inf).reshape(-1)
        next_idx = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[next_idx]):
            break
        selected.reshape(-1)[next_idx] = True

    out = np.zeros_like(score_grid, dtype=np.float64)
    out[selected] = 1.0
    return out


def project_scores_to_topk_binary_torch(scores: torch.Tensor, k: int) -> torch.Tensor:
    flat = scores.reshape(scores.shape[0], -1)
    out = torch.zeros_like(flat)
    if k <= 0:
        return out.reshape_as(scores)
    k = min(k, flat.shape[1])
    top_idx = torch.topk(flat, k=k, dim=1).indices
    out.scatter_(1, top_idx, 1.0)
    return out.reshape_as(scores)


def apply_projection_numpy(
    x: np.ndarray, beta: float, eta: float, hard_binarize: bool
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if beta > 0:
        num = np.tanh(beta * eta) + np.tanh(beta * (out - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        out = num / max(den, 1e-12)
    if hard_binarize:
        out = (out >= eta).astype(np.float64)
    return out


def apply_projection_torch(
    x: torch.Tensor, beta: float, eta: float, hard_binarize: bool
) -> torch.Tensor:
    out = x
    if beta > 0:
        num = torch.tanh(
            torch.tensor(beta * eta, device=x.device, dtype=x.dtype)
        ) + torch.tanh(beta * (out - eta))
        den = torch.tanh(
            torch.tensor(beta * eta, device=x.device, dtype=x.dtype)
        ) + torch.tanh(torch.tensor(beta * (1.0 - eta), device=x.device, dtype=x.dtype))
        out = num / torch.clamp(den, min=1e-12)
    if hard_binarize:
        out = (out >= eta).to(out.dtype)
    return out


def parse_ladder_sequence(specs: list[str]) -> tuple[tuple[LadderKind, float], ...]:
    if not specs:
        return ()
    parsed: list[tuple[LadderKind, float]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid ladder spec '{spec}'. Expected format kind:value, e.g. volume:0.52"
            )
        kind_raw, value_raw = spec.split(":", 1)
        kind = kind_raw.strip().lower()
        if kind not in {
            "volume",
            "compliance",
            "roughness",
            "connectivity",
            "diversity",
        }:
            raise ValueError(
                f"Invalid ladder kind '{kind_raw}'. Expected one of volume, compliance, roughness, connectivity, diversity"
            )
        parsed.append((kind, float(value_raw)))
    return tuple(parsed)


def parse_levels_ladder_specs(specs: list[str]) -> list[Rung]:
    """Parse official Levels.ladder specs like compliance:130,110,100."""
    allowed = {"volume", "roughness", "compliance", "connectivity", "diversity"}
    rungs: list[Rung] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) == 2:
            name, thresholds_raw = parts
            direction = "min"
        elif len(parts) == 3:
            name, direction, thresholds_raw = parts
        else:
            raise ValueError(
                "Invalid --levels_ladder spec. Expected name:v1,v2 or name:min:v1,v2"
            )
        if name not in allowed:
            raise ValueError(
                f"Invalid ladder objective '{name}'. Expected one of {sorted(allowed)}"
            )
        if direction not in {"min", "max"}:
            raise ValueError(f"Invalid ladder direction '{direction}'")
        thresholds = [float(value) for value in thresholds_raw.split(",") if value]
        if not thresholds:
            raise ValueError(f"No thresholds provided for ladder objective '{name}'")
        if direction == "min":
            rungs.append(Rung.minimize(name, thresholds))
        else:
            rungs.append(Rung.maximize(name, thresholds))
    return rungs


def get_level_names(
    volume_ladder: list[float],
    compliance_ladder: list[float],
    roughness_ladder: list[float],
    connectivity_ladder: list[float],
    diversity_ladder: list[float],
    connectivity_max: float | None,
    ladder_sequence: tuple[tuple[LadderKind, float], ...],
) -> list[str]:
    names: list[str] = []
    if ladder_sequence:
        for kind, bound in ladder_sequence:
            names.append(f"{kind}_violation_le_{bound:g}")
    else:
        names.extend(f"volume_violation_le_{bound:g}" for bound in volume_ladder)
        names.extend(
            f"compliance_violation_le_{bound:g}" for bound in compliance_ladder
        )
        names.extend(f"roughness_violation_le_{bound:g}" for bound in roughness_ladder)
        names.extend(
            f"connectivity_violation_le_{bound:g}" for bound in connectivity_ladder
        )
        names.extend(f"diversity_violation_ge_{bound:g}" for bound in diversity_ladder)
    if connectivity_max is not None:
        names.append("connectivity_violation")
    names.extend(["volume_violation", "roughness_violation", "compliance"])
    return names


def ensure_torchfem_importable(torchfem_src: str | None) -> None:
    try:
        import torchfem  # noqa: F401

        return
    except ImportError:
        pass

    if torchfem_src is None:
        raise ImportError(
            "torch-fem is not installed. Pass --torchfem_src pointing to a torch-fem src directory."
        )

    src_path = Path(torchfem_src).expanduser().resolve()
    if not src_path.exists():
        raise ValueError(f"torchfem_src does not exist: {src_path}")

    if "pyvista" not in sys.modules:
        pyvista_stub = types.ModuleType("pyvista")
        pyvista_stub.DataSet = object
        pyvista_stub.Plotter = object
        sys.modules["pyvista"] = pyvista_stub

    sys.path.insert(0, str(src_path))
    import torchfem  # noqa: F401


class FEMCantileverEvaluator:
    """2D linear-elasticity cantilever on a regular quad mesh.

    The evaluator is intentionally array-only so GFog remains separated from the
    mechanics backend. The generator emits unconstrained logits; the black-box
    evaluator maps them to physical densities with a sigmoid, then filters,
    assembles a sparse global stiffness matrix, solves the reduced system, and
    returns plain array values.
    """

    def __init__(self, config: FEMConfig) -> None:
        self.config = config
        self._fem_executor: ThreadPoolExecutor | None = None
        self._fem_executor_workers = 0
        self.requires_connectivity = self._requires_connectivity_objective(config)
        self.nelx = config.grid_width
        self.nely = config.grid_height
        self._configure_code_shape()
        self.tiny_decoder: Any | None = None
        self.nelems = self.nelx * self.nely
        self.nnodes = (self.nelx + 1) * (self.nely + 1)
        self.ndof = 2 * self.nnodes
        self.ke = self._element_stiffness(config.poisson_ratio)
        self.edof_mat = self._build_edof_mat()
        self.iK = np.kron(self.edof_mat, np.ones((8, 1), dtype=np.int32)).ravel()
        self.jK = np.kron(self.edof_mat, np.ones((1, 8), dtype=np.int32)).ravel()
        self.fixed_dofs, self.free_dofs = self._build_boundary_conditions()
        self.n_free_dofs = int(len(self.free_dofs))
        self._precompute_free_stiffness_entries()
        self._mf_cache_key: tuple[str, torch.dtype] | None = None
        self._mf_edof: torch.Tensor | None = None
        self._mf_free_dofs: torch.Tensor | None = None
        self._mf_ke: torch.Tensor | None = None
        self._mf_force_free: torch.Tensor | None = None
        self.last_matrix_free_cg_iterations: np.ndarray | None = None
        self.last_matrix_free_cg_relative_residuals: np.ndarray | None = None
        self.force_cases = self._active_force_cases()
        self.forces = tuple(
            self._build_force_vector_for_case(load_case)
            for load_case in self.force_cases
        )
        self.force = self.forces[0]
        self.force_frees = tuple(force[self.free_dofs] for force in self.forces)
        self.force_free = self.force_frees[0]
        self.filter_offsets, self.filter_weights = self._build_filter_kernel()
        self.sorted_material_values = fixed_sorted_material_values_numpy(
            self.nelems,
            self.config.volume_max,
            profile=self.config.sorted_material_profile,
            steepness=self.config.sorted_material_steepness,
        )
        self.diversity_reference_designs: np.ndarray | None = None
        self.solid_compliance = self._compute_solid_compliance()

    def set_density_filter_radius(self, radius: int) -> None:
        """Update the density filter kernel used by subsequent decodes."""
        if radius < 0:
            raise ValueError(
                f"density_filter_radius must be non-negative, got {radius}"
            )
        if radius == self.config.density_filter_radius:
            return
        self.config.density_filter_radius = radius
        self.filter_offsets, self.filter_weights = self._build_filter_kernel()
        self.solid_compliance = self._compute_solid_compliance()

    def close(self) -> None:
        """Release persistent FEM worker resources."""
        if self._fem_executor is not None:
            self._fem_executor.shutdown(wait=True)
            self._fem_executor = None
            self._fem_executor_workers = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_fem_executor(self) -> ThreadPoolExecutor:
        if (
            self._fem_executor is None
            or self._fem_executor_workers != self.config.fem_workers
        ):
            self.close()
            self._fem_executor = ThreadPoolExecutor(max_workers=self.config.fem_workers)
            self._fem_executor_workers = self.config.fem_workers
        return self._fem_executor

    def _requires_connectivity_objective(self, config: FEMConfig) -> bool:
        if config.connectivity_max is not None or bool(config.connectivity_ladder):
            return True
        if any(kind == "connectivity" for kind, _bound in config.ladder_sequence):
            return True
        return (
            config.use_levels_ladder
            and "connectivity" in config.levels_ladder_objectives
        )

    def _configure_code_shape(self) -> None:
        if self.config.encoding == "tiny_decoder":
            self.code_width = self.config.tiny_decoder_latent_width
            self.code_height = self.config.tiny_decoder_latent_height
            self.code_channels = self.config.tiny_decoder_latent_channels
        elif self.config.encoding == "bar_primitives":
            self.code_width = 5 * self.config.bar_count
            self.code_height = 1
            self.code_channels = 1
        else:
            self.code_width = self.config.coarse_grid_width or self.config.grid_width
            self.code_height = self.config.coarse_grid_height or self.config.grid_height
            self.code_channels = 1
        self.code_dim = self.code_channels * self.code_width * self.code_height

    def _load_tiny_decoder(self) -> Any:
        if self.tiny_decoder is not None:
            return self.tiny_decoder
        try:
            from diffusers import AutoencoderTiny
        except ImportError as exc:
            raise ImportError(
                "--encoding tiny_decoder requires diffusers. Install it and make "
                "the configured --tiny_decoder_model available locally or via Hugging Face."
            ) from exc

        decoder = AutoencoderTiny.from_pretrained(self.config.tiny_decoder_model)
        decoder.eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        self.tiny_decoder = decoder.to("cpu")
        return self.tiny_decoder

    def _decode_tiny_decoder_numpy(self, x_np: np.ndarray) -> np.ndarray:
        latent = torch.from_numpy(x_np.reshape(-1, self.code_dim)).to(torch.float32)
        latent = torch.tanh(latent) * self.config.tiny_decoder_latent_scale
        latent = latent.reshape(
            -1,
            self.config.tiny_decoder_latent_channels,
            self.config.tiny_decoder_latent_height,
            self.config.tiny_decoder_latent_width,
        )
        decoder = self._load_tiny_decoder()
        with torch.no_grad():
            decoded = decoder.decode(latent)
            image = decoded.sample if hasattr(decoded, "sample") else decoded
            image = torch.clamp((image + 1.0) * 0.5, 0.0, 1.0)
            gray = image.mean(dim=1, keepdim=True)
            resized = F.interpolate(
                gray,
                size=(self.nely, self.nelx),
                mode="bilinear",
                align_corners=False,
            )
        scores = resized[:, 0].reshape(-1, self.nely * self.nelx).cpu().numpy()
        k = int(round(self.config.volume_max * self.nely * self.nelx))
        return np.stack(
            [project_scores_to_topk_binary_numpy(sample, k) for sample in scores],
            axis=0,
        ).reshape(-1, self.nely, self.nelx)

    def _element_stiffness(self, nu: float) -> np.ndarray:
        a11 = np.array(
            [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]],
            dtype=np.float64,
        )
        a12 = np.array(
            [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]],
            dtype=np.float64,
        )
        b11 = np.array(
            [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]],
            dtype=np.float64,
        )
        b12 = np.array(
            [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]],
            dtype=np.float64,
        )
        return (
            np.block([[a11, a12], [a12.T, a11]])
            + nu * np.block([[b11, b12], [b12.T, b11]])
        ) / (24.0 * (1.0 - nu**2))

    def _build_edof_mat(self) -> np.ndarray:
        edof = np.zeros((self.nelems, 8), dtype=np.int32)
        elem = 0
        for row in range(self.nely):
            for col in range(self.nelx):
                n1 = row * (self.nelx + 1) + col
                n2 = n1 + 1
                n4 = n1 + (self.nelx + 1)
                n3 = n4 + 1
                edof[elem] = np.array(
                    [
                        2 * n1,
                        2 * n1 + 1,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n3,
                        2 * n3 + 1,
                        2 * n4,
                        2 * n4 + 1,
                    ],
                    dtype=np.int32,
                )
                elem += 1
        return edof

    def _build_boundary_conditions(self) -> tuple[np.ndarray, np.ndarray]:
        fixed = []
        for row in range(self.nely + 1):
            node = row * (self.nelx + 1)
            fixed.extend([2 * node, 2 * node + 1])
        fixed_dofs = np.asarray(sorted(fixed), dtype=np.int32)
        all_dofs = np.arange(self.ndof, dtype=np.int32)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        return fixed_dofs, free_dofs

    def _precompute_free_stiffness_entries(self) -> None:
        """Precompute reduced stiffness entries for repeated compliance solves."""
        free_index = np.full(self.ndof, -1, dtype=np.int32)
        free_index[self.free_dofs] = np.arange(self.n_free_dofs, dtype=np.int32)

        reduced_i = free_index[self.iK]
        reduced_j = free_index[self.jK]
        free_mask = (reduced_i >= 0) & (reduced_j >= 0)

        entry_elements = np.repeat(np.arange(self.nelems, dtype=np.int32), 64)
        entry_ke = np.tile(self.ke.ravel(), self.nelems)

        self.iK_free = reduced_i[free_mask]
        self.jK_free = reduced_j[free_mask]
        self.free_entry_elements = entry_elements[free_mask]
        self.free_entry_ke = entry_ke[free_mask]

    def _active_force_cases(self) -> tuple[LoadCase, ...]:
        if not self.config.robust_load_cases:
            return (self.config.load_case,)
        force_cases = []
        seen = set()
        for load_case in self.config.robust_load_cases:
            if load_case in seen:
                continue
            force_cases.append(load_case)
            seen.add(load_case)
        if not force_cases:
            raise ValueError("--robust_load_cases must contain at least one load case")
        return tuple(force_cases)

    def _build_force_vector(self) -> np.ndarray:
        return self._build_force_vector_for_case(self.config.load_case)

    def _build_force_vector_for_case(self, load_case: LoadCase) -> np.ndarray:
        force = np.zeros(self.ndof, dtype=np.float64)
        if load_case == "tom_two_patches":
            return self._build_tom_two_patch_force_vector(force)
        if load_case == "center_point":
            return self._add_right_edge_point_load(force, self.nely // 2)
        if load_case == "right_top_point":
            return self._add_right_edge_point_load(force, self.nely)
        if load_case == "right_bottom_point":
            return self._add_right_edge_point_load(force, 0)
        if load_case == "right_two_points":
            force = self._add_right_edge_point_load(
                force,
                max(0, int(round(0.25 * self.nely))),
                scale=0.5,
            )
            return self._add_right_edge_point_load(
                force,
                min(self.nely, int(round(0.75 * self.nely))),
                scale=0.5,
            )
        if load_case == "right_edge_uniform":
            return self._build_right_edge_uniform_force_vector(force)
        if load_case == "right_edge_shear":
            return self._build_right_edge_shear_force_vector(force)
        raise ValueError(f"Unknown load_case: {load_case}")

    def _add_right_edge_point_load(
        self,
        force: np.ndarray,
        load_row: int,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        load_row = min(max(load_row, 0), self.nely)
        load_node = load_row * (self.nelx + 1) + self.nelx
        force[2 * load_node + 1] += -scale * self.config.load_scale
        return force

    def _build_right_edge_uniform_force_vector(self, force: np.ndarray) -> np.ndarray:
        """Uniform downward traction on the full free right edge."""
        for row in range(self.nely):
            lower_node = row * (self.nelx + 1) + self.nelx
            upper_node = (row + 1) * (self.nelx + 1) + self.nelx
            nodal_force = -0.5 * self.config.load_scale / self.nely
            force[2 * lower_node + 1] += nodal_force
            force[2 * upper_node + 1] += nodal_force
        return force

    def _build_right_edge_shear_force_vector(self, force: np.ndarray) -> np.ndarray:
        """Horizontal shear at the center of the right edge."""
        load_row = self.nely // 2
        load_node = load_row * (self.nelx + 1) + self.nelx
        force[2 * load_node] = self.config.load_scale
        return force

    def _build_tom_two_patch_force_vector(self, force: np.ndarray) -> np.ndarray:
        """Consistent nodal loads for TOM's two right-edge traction patches."""
        edge_length = self.config.domain_height / self.nely
        for row in range(self.nely):
            y0 = row / self.nely
            y1 = (row + 1) / self.nely
            yc = 0.5 * (y0 + y1)
            in_patch = (0.1 < yc < 0.2) or (0.8 < yc < 0.9)
            if not in_patch:
                continue
            lower_node = row * (self.nelx + 1) + self.nelx
            upper_node = (row + 1) * (self.nelx + 1) + self.nelx
            nodal_force = -0.5 * self.config.load_scale * edge_length
            force[2 * lower_node + 1] += nodal_force
            force[2 * upper_node + 1] += nodal_force
        return force

    def _build_filter_kernel(self) -> tuple[list[tuple[int, int]], np.ndarray]:
        radius = self.config.density_filter_radius
        offsets: list[tuple[int, int]] = []
        weights: list[float] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                dist = float(np.sqrt(dr * dr + dc * dc))
                weight = max(radius + 1 - dist, 0.0)
                if weight > 0.0:
                    offsets.append((dr, dc))
                    weights.append(weight)
        return offsets, np.asarray(weights, dtype=np.float64)

    def _apply_density_filter(self, sample: np.ndarray) -> np.ndarray:
        if self.config.density_filter_radius <= 0:
            return sample
        filtered = np.zeros_like(sample, dtype=np.float64)
        weight_sum = np.zeros_like(sample, dtype=np.float64)
        for (dr, dc), weight in zip(
            self.filter_offsets, self.filter_weights, strict=False
        ):
            src_r0 = max(0, -dr)
            src_r1 = sample.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = sample.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            filtered[dst_r0:dst_r1, dst_c0:dst_c1] += (
                weight * sample[src_r0:src_r1, src_c0:src_c1]
            )
            weight_sum[dst_r0:dst_r1, dst_c0:dst_c1] += weight
        return filtered / np.maximum(weight_sum, 1e-12)

    def _apply_density_filter_torch(self, samples: torch.Tensor) -> torch.Tensor:
        if self.config.density_filter_radius <= 0:
            return samples

        radius = self.config.density_filter_radius
        kernel_size = 2 * radius + 1
        kernel = torch.zeros(
            (1, 1, kernel_size, kernel_size),
            device=samples.device,
            dtype=samples.dtype,
        )
        for (dr, dc), weight in zip(
            self.filter_offsets, self.filter_weights, strict=False
        ):
            kernel[0, 0, dr + radius, dc + radius] = float(weight)

        x = samples.unsqueeze(1)
        filtered = F.conv2d(x, kernel, padding=radius)
        weight_sum = F.conv2d(torch.ones_like(x), kernel, padding=radius)
        return (filtered / torch.clamp(weight_sum, min=1e-12)).squeeze(1)

    def _assemble_stiffness(self, density_phys: np.ndarray) -> sp.csc_matrix:
        penalized = self.config.e_min + (
            density_phys.ravel(order="C") ** self.config.simp_p
        ) * (self.config.e_max - self.config.e_min)
        sK = (self.ke.ravel()[None, :] * penalized[:, None]).ravel()
        K = sp.coo_matrix(
            (sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)
        ).tocsc()
        return (K + K.T) * 0.5

    def _assemble_reduced_stiffness(self, density_phys: np.ndarray) -> sp.csc_matrix:
        penalized = self.config.e_min + (
            density_phys.ravel(order="C") ** self.config.simp_p
        ) * (self.config.e_max - self.config.e_min)
        sK = self.free_entry_ke * penalized[self.free_entry_elements]
        return sp.coo_matrix(
            (sK, (self.iK_free, self.jK_free)),
            shape=(self.n_free_dofs, self.n_free_dofs),
        ).tocsc()

    def _solve_compliance_for_force(
        self,
        density_phys: np.ndarray,
        force_free: np.ndarray,
    ) -> float:
        K_ff = self._assemble_reduced_stiffness(density_phys)
        u_f = spla.spsolve(K_ff, force_free)
        compliance = float(force_free @ u_f)
        return compliance

    def _aggregate_compliances(self, compliances: list[float]) -> float:
        if not compliances:
            raise ValueError("Cannot aggregate an empty compliance list")
        values = np.asarray(compliances, dtype=np.float64)
        if self.config.robust_load_aggregate == "max":
            return float(values.max())
        if self.config.robust_load_aggregate == "mean":
            return float(values.mean())
        if self.config.robust_load_aggregate == "cvar":
            count = max(
                1,
                int(math.ceil(self.config.robust_load_cvar_frac * len(values))),
            )
            return float(np.sort(values)[-count:].mean())
        raise ValueError(
            f"Unknown robust_load_aggregate: {self.config.robust_load_aggregate}"
        )

    def _solve_compliance(self, density_phys: np.ndarray) -> float:
        if len(self.force_frees) == 1:
            return self._solve_compliance_for_force(density_phys, self.force_free)

        K_ff = self._assemble_reduced_stiffness(density_phys)
        rhs = np.column_stack(self.force_frees)
        displacements = spla.spsolve(K_ff, rhs)
        if displacements.ndim == 1:
            displacements = displacements[:, None]
        compliances = np.sum(rhs * displacements, axis=0).astype(np.float64).tolist()
        return self._aggregate_compliances(compliances)

    def _ensure_matrix_free_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = matrix_free_cg_torch_dtype(self.config.matrix_free_cg_dtype)
        device = torch.device(self.config.matrix_free_cg_device)
        key = (str(device), dtype)
        if self._mf_cache_key != key:
            self._mf_edof = torch.as_tensor(
                self.edof_mat.astype(np.int64),
                device=device,
            )
            self._mf_free_dofs = torch.as_tensor(
                self.free_dofs.astype(np.int64),
                device=device,
            )
            self._mf_ke = torch.as_tensor(self.ke, device=device, dtype=dtype)
            self._mf_force_free = torch.as_tensor(
                self.force_free,
                device=device,
                dtype=dtype,
            )
            self._mf_cache_key = key
        if (
            self._mf_edof is None
            or self._mf_free_dofs is None
            or self._mf_ke is None
            or self._mf_force_free is None
        ):
            raise RuntimeError("matrix-free tensor cache was not initialized")
        return self._mf_edof, self._mf_free_dofs, self._mf_ke, self._mf_force_free

    def _matrix_free_matvec(
        self,
        x_free: torch.Tensor,
        moduli: torch.Tensor,
    ) -> torch.Tensor:
        edof, free_dofs, ke, _force_free = self._ensure_matrix_free_tensors()
        batch_size = x_free.shape[0]
        x_global = torch.zeros(
            (batch_size, self.ndof),
            device=x_free.device,
            dtype=x_free.dtype,
        )
        x_global[:, free_dofs] = x_free
        x_elem = x_global[:, edof]
        kx_elem = torch.einsum("ij,bej->bei", ke, x_elem)
        kx_elem = kx_elem * moduli[:, :, None]
        out_global = torch.zeros_like(x_global)
        scatter_index = edof.reshape(1, -1).expand(batch_size, -1)
        out_global.scatter_add_(1, scatter_index, kx_elem.reshape(batch_size, -1))
        return out_global[:, free_dofs]

    def _matrix_free_jacobi_diagonal(self, moduli: torch.Tensor) -> torch.Tensor:
        edof, free_dofs, ke, _force_free = self._ensure_matrix_free_tensors()
        batch_size = moduli.shape[0]
        elem_diag = torch.diagonal(ke).reshape(1, 1, -1) * moduli[:, :, None]
        diag_global = torch.zeros(
            (batch_size, self.ndof),
            device=moduli.device,
            dtype=moduli.dtype,
        )
        scatter_index = edof.reshape(1, -1).expand(batch_size, -1)
        diag_global.scatter_add_(1, scatter_index, elem_diag.reshape(batch_size, -1))
        return torch.clamp(diag_global[:, free_dofs], min=1e-30)

    def _solve_compliance_matrix_free_batch(self, samples: np.ndarray) -> np.ndarray:
        if len(self.force_frees) != 1:
            raise ValueError(
                "--compliance_solver matrix_free_cg currently supports one load case"
            )
        _edof, _free_dofs, _ke, force_free = self._ensure_matrix_free_tensors()
        with torch.no_grad():
            densities = torch.as_tensor(
                np.asarray(samples, dtype=np.float64),
                device=force_free.device,
                dtype=force_free.dtype,
            ).reshape(-1, self.nely, self.nelx)
            batch_size = densities.shape[0]
            moduli = self.config.e_min + (
                densities.reshape(batch_size, -1).pow(self.config.simp_p)
                * (self.config.e_max - self.config.e_min)
            )
            b = force_free.reshape(1, -1).expand(batch_size, -1)
            b_norm = torch.clamp(torch.linalg.vector_norm(b, dim=1), min=1e-30)
            x = torch.zeros_like(b)
            r = b.clone()
            diag = self._matrix_free_jacobi_diagonal(moduli)
            z = r / diag
            p = z.clone()
            rz = torch.sum(r * z, dim=1)
            rel = torch.linalg.vector_norm(r, dim=1) / b_norm
            iterations = torch.zeros(
                batch_size,
                device=force_free.device,
                dtype=torch.int64,
            )
            active = rel > self.config.matrix_free_cg_tol

            for step in range(1, self.config.matrix_free_cg_max_iter + 1):
                if not bool(torch.any(active).item()):
                    break
                if p.device.type == "mps":
                    ap = self._matrix_free_matvec(p, moduli)
                else:
                    active_idx = torch.nonzero(active, as_tuple=False).flatten()
                    ap = torch.zeros_like(p)
                    ap_active = self._matrix_free_matvec(
                        p.index_select(0, active_idx),
                        moduli.index_select(0, active_idx),
                    )
                    ap.index_copy_(0, active_idx, ap_active)
                denom = torch.clamp(torch.sum(p * ap, dim=1), min=1e-30)
                alpha = torch.where(active, rz / denom, torch.zeros_like(rz))
                x_next = x + alpha[:, None] * p
                r_next = r - alpha[:, None] * ap
                z_next = r_next / diag
                rz_next = torch.sum(r_next * z_next, dim=1)
                beta = torch.where(
                    active,
                    rz_next / torch.clamp(rz, min=1e-30),
                    torch.zeros_like(rz),
                )
                p_next = z_next + beta[:, None] * p

                active_col = active[:, None]
                x = torch.where(active_col, x_next, x)
                r = torch.where(active_col, r_next, r)
                z = torch.where(active_col, z_next, z)
                p = torch.where(active_col, p_next, p)
                rz = torch.where(active, rz_next, rz)
                rel = torch.linalg.vector_norm(r, dim=1) / b_norm
                newly_converged = active & (rel <= self.config.matrix_free_cg_tol)
                iterations = torch.where(
                    newly_converged,
                    torch.full_like(iterations, step),
                    iterations,
                )
                active = active & (rel > self.config.matrix_free_cg_tol)

            iterations = torch.where(
                iterations == 0,
                torch.full_like(iterations, self.config.matrix_free_cg_max_iter),
                iterations,
            )
            compliances = torch.sum(b * x, dim=1)
            self.last_matrix_free_cg_iterations = iterations.cpu().numpy()
            self.last_matrix_free_cg_relative_residuals = rel.cpu().numpy()
            return compliances.cpu().numpy().astype(np.float64, copy=False)

    def _finalize_density_sample(self, sample_raw: np.ndarray) -> np.ndarray:
        sample = self._apply_density_filter(sample_raw.astype(np.float64, copy=False))
        return apply_projection_numpy(
            sample,
            beta=self.config.projection_beta,
            eta=self.config.projection_eta,
            hard_binarize=self.config.hard_binarize,
        )

    def decode_designs_numpy(self, x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float32)
        binhead_scores: np.ndarray | None = None

        if self.config.encoding == "direct":
            x_phys = decode_design_logits_numpy(x_np).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "soft_volume":
            logits = x_np.reshape(-1, self.nely, self.nelx)
            x_phys = sigmoid_with_target_mean_numpy(
                logits,
                target_mean=self.config.volume_max,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "coarse":
            x_full = expand_design_code_numpy(
                x_np,
                coarse_height=self.code_height,
                coarse_width=self.code_width,
                full_height=self.nely,
                full_width=self.nelx,
            )
            x_phys = decode_design_logits_numpy(x_full).reshape(
                -1, self.nely, self.nelx
            )
        elif self.config.encoding == "binary_coarse":
            coarse_scores = (
                F.softplus(torch.from_numpy(x_np.reshape(-1, self.code_dim)))
                .cpu()
                .numpy()
            )
            coarse_binary = np.stack(
                [
                    project_positive_vector_to_binary_numpy(sample)
                    for sample in coarse_scores
                ],
                axis=0,
            ).reshape(-1, self.code_height, self.code_width)
            x_phys = expand_design_code_numpy(
                coarse_binary,
                coarse_height=self.code_height,
                coarse_width=self.code_width,
                full_height=self.nely,
                full_width=self.nelx,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "topk_volume":
            final_scores = x_np.reshape(-1, self.nely * self.nelx)
            binhead_scores = final_scores.reshape(-1, self.nely, self.nelx)
            k = int(round(self.config.volume_max * self.nely * self.nelx))
            x_phys = np.stack(
                [
                    project_scores_to_topk_binary_numpy(sample, k)
                    for sample in final_scores
                ],
                axis=0,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "sorted_material":
            final_scores = x_np.reshape(-1, self.nely * self.nelx)
            binhead_scores = final_scores.reshape(-1, self.nely, self.nelx)
            x_phys = np.stack(
                [
                    project_scores_to_fixed_sorted_material_numpy(
                        sample, self.sorted_material_values
                    )
                    for sample in final_scores
                ],
                axis=0,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "tiny_decoder":
            x_phys = self._decode_tiny_decoder_numpy(x_np)
        elif self.config.encoding == "coarse_topk_volume":
            coarse_scores = x_np.reshape(-1, self.code_height, self.code_width)
            upsampled_scores = expand_design_code_numpy(
                coarse_scores,
                coarse_height=self.code_height,
                coarse_width=self.code_width,
                full_height=self.nely,
                full_width=self.nelx,
            )
            final_scores = upsampled_scores.reshape(-1, self.nely * self.nelx)
            binhead_scores = final_scores.reshape(-1, self.nely, self.nelx)
            k = int(round(self.config.volume_max * self.nely * self.nelx))
            x_phys = np.stack(
                [
                    project_scores_to_topk_binary_numpy(sample, k)
                    for sample in final_scores
                ],
                axis=0,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "coarse_residual":
            coarse_dim = self.code_height * self.code_width
            full_dim = self.nely * self.nelx
            if x_np.shape[-1] != coarse_dim + full_dim:
                raise ValueError(
                    "coarse_residual encoding expects output_dim = coarse_dim + full_dim "
                    f"({coarse_dim} + {full_dim}), got {x_np.shape[-1]}"
                )
            coarse_logits = x_np[:, :coarse_dim].reshape(
                -1, self.code_height, self.code_width
            )
            residual_logits = x_np[:, coarse_dim:].reshape(-1, self.nely, self.nelx)
            coarse_up = expand_design_code_numpy(
                coarse_logits,
                coarse_height=self.code_height,
                coarse_width=self.code_width,
                full_height=self.nely,
                full_width=self.nelx,
            )
            combined_logits = coarse_up + self.config.residual_scale * residual_logits
            x_phys = decode_design_logits_numpy(combined_logits).reshape(
                -1, self.nely, self.nelx
            )
        elif self.config.encoding == "bar_primitives":
            expected_dim = 5 * self.config.bar_count
            if x_np.shape[-1] != expected_dim:
                raise ValueError(
                    f"bar_primitives expects output_dim={expected_dim}, got {x_np.shape[-1]}"
                )
            x_phys = rasterize_bar_primitives_numpy(
                x_np.reshape(-1, self.config.bar_count, 5),
                nely=self.nely,
                nelx=self.nelx,
                width_min=self.config.bar_width_min,
                width_max=self.config.bar_width_max,
                edge_softness=self.config.bar_edge_softness,
            )
        else:
            raise ValueError(f"Unknown encoding mode: {self.config.encoding}")

        decoded = []
        target_count = int(round(self.config.volume_max * self.nely * self.nelx))
        for idx, sample_raw in enumerate(x_phys.reshape(-1, self.nely, self.nelx)):
            if self.config.binhead_connect_support and binhead_scores is not None:
                sample_raw = connect_support_and_refill_numpy(
                    sample_raw,
                    binhead_scores[idx],
                    target_count=target_count,
                )
            decoded.append(self._finalize_density_sample(sample_raw))
        return np.asarray(decoded, dtype=np.float32)

    def decode_designs_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiably decode generator output to physical density fields."""
        if self.config.hard_binarize:
            raise ValueError("--train_on_decoded does not support --hard_binarize")

        if self.config.encoding == "direct":
            x_phys = decode_design_logits_torch(x).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "soft_volume":
            logits = x.reshape(-1, self.nely, self.nelx)
            x_phys = sigmoid_with_target_mean_torch(
                logits,
                target_mean=self.config.volume_max,
            ).reshape(-1, self.nely, self.nelx)
        elif self.config.encoding == "coarse":
            x_full = F.interpolate(
                x.reshape(-1, 1, self.code_height, self.code_width),
                size=(self.nely, self.nelx),
                mode="bilinear",
                align_corners=False,
            )
            x_phys = decode_design_logits_torch(x_full[:, 0]).reshape(
                -1, self.nely, self.nelx
            )
        elif self.config.encoding == "coarse_residual":
            coarse_dim = self.code_height * self.code_width
            full_dim = self.nely * self.nelx
            if x.shape[-1] != coarse_dim + full_dim:
                raise ValueError(
                    "coarse_residual encoding expects output_dim = coarse_dim + full_dim "
                    f"({coarse_dim} + {full_dim}), got {x.shape[-1]}"
                )
            coarse_logits = x[:, :coarse_dim].reshape(
                -1, 1, self.code_height, self.code_width
            )
            residual_logits = x[:, coarse_dim:].reshape(-1, self.nely, self.nelx)
            coarse_up = F.interpolate(
                coarse_logits,
                size=(self.nely, self.nelx),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            x_phys = decode_design_logits_torch(
                coarse_up + self.config.residual_scale * residual_logits
            ).reshape(-1, self.nely, self.nelx)
        else:
            raise ValueError(
                "--train_on_decoded only supports differentiable encodings: "
                "direct, soft_volume, coarse, coarse_residual"
            )

        x_phys = self._apply_density_filter_torch(x_phys)
        return apply_projection_torch(
            x_phys,
            beta=self.config.projection_beta,
            eta=self.config.projection_eta,
            hard_binarize=False,
        )

    def _compute_solid_compliance(self) -> float:
        solid = np.ones((self.nely, self.nelx), dtype=np.float64)
        solid = self._apply_density_filter(solid)
        solid = apply_projection_numpy(
            solid,
            beta=self.config.projection_beta,
            eta=self.config.projection_eta,
            hard_binarize=self.config.hard_binarize,
        )
        return self._solve_compliance(solid)

    def disconnected_solid_fraction(self, sample: np.ndarray) -> float:
        solid = sample >= self.config.projection_eta
        total_solid = int(np.count_nonzero(solid))
        if total_solid == 0:
            return 1.0

        visited = np.zeros_like(solid, dtype=bool)
        stack: list[tuple[int, int]] = [
            (row, 0) for row in range(self.nely) if bool(solid[row, 0])
        ]
        for row, col in stack:
            visited[row, col] = True

        connected = 0
        while stack:
            row, col = stack.pop()
            connected += 1
            for nr, nc in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if (
                    0 <= nr < self.nely
                    and 0 <= nc < self.nelx
                    and bool(solid[nr, nc])
                    and not bool(visited[nr, nc])
                ):
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        return float(max(total_solid - connected, 0) / total_solid)

    def _boundary_points_numpy(self, sample: np.ndarray) -> np.ndarray:
        solid = sample >= self.config.projection_eta
        if not np.any(solid):
            return np.asarray([[0.5, 0.5]], dtype=np.float32)

        padded = np.pad(solid, 1, constant_values=False)
        boundary = solid & (
            ~padded[:-2, 1:-1]
            | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2]
            | ~padded[1:-1, 2:]
        )
        points = np.argwhere(boundary)
        if points.size == 0:
            points = np.argwhere(solid)
        max_points = self.config.diversity_chamfer_max_points
        if points.shape[0] > max_points:
            idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
            points = points[idx]
        coords = np.empty((points.shape[0], 2), dtype=np.float32)
        coords[:, 0] = (points[:, 1].astype(np.float32) + 0.5) / float(self.nelx)
        coords[:, 1] = (points[:, 0].astype(np.float32) + 0.5) / float(self.nely)
        return coords

    @staticmethod
    def _chamfer_distance_numpy(a: np.ndarray, b: np.ndarray) -> float:
        diff = a[:, None, :] - b[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        return float(
            np.sqrt(np.min(dist2, axis=1)).mean()
            + np.sqrt(np.min(dist2, axis=0)).mean()
        )

    def boundary_chamfer_diversity(self, sample: np.ndarray) -> float:
        refs = self.diversity_reference_designs
        if refs is None or len(refs) == 0:
            return float("inf")
        sample_points = self._boundary_points_numpy(sample)
        best = float("inf")
        for reference in refs:
            ref_points = self._boundary_points_numpy(reference)
            best = min(best, self._chamfer_distance_numpy(sample_points, ref_points))
        return best

    def density_objectives(
        self, sample: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        volume = float(sample.mean())
        dx = float(np.abs(sample[:, 1:] - sample[:, :-1]).mean())
        dy = float(np.abs(sample[1:, :] - sample[:-1, :]).mean())
        roughness = 0.5 * (dx + dy)
        connectivity = (
            self.disconnected_solid_fraction(sample)
            if self.requires_connectivity
            else 0.0
        )
        diversity = (
            self.boundary_chamfer_diversity(sample)
            if self.config.diversity_ladder
            or any(kind == "diversity" for kind, _bound in self.config.ladder_sequence)
            or (
                self.config.use_levels_ladder
                and "diversity" in self.config.levels_ladder_objectives
            )
            else float("inf")
        )
        compliance = self._solve_compliance(sample)
        return volume, roughness, connectivity, diversity, compliance

    def removal_ladder_objectives(self, scores: np.ndarray) -> list[float]:
        flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if flat_scores.size != self.nelems:
            raise ValueError(
                "removal ladder expects one priority score per element, got "
                f"{flat_scores.size} scores for {self.nelems} elements"
            )
        volumes = self.config.removal_ladder_volumes
        targets = self.config.removal_ladder_compliances
        if not volumes:
            raise ValueError("removal_ladder_objectives requires configured volumes")

        compliances = []
        connectivities = []
        for volume_fraction in volumes:
            k = int(round(volume_fraction * self.nelems))
            sample_raw = project_scores_to_topk_binary_numpy(flat_scores, k).reshape(
                self.nely,
                self.nelx,
            )
            if self.config.binhead_connect_support:
                sample_raw = connect_support_and_refill_numpy(
                    sample_raw,
                    flat_scores.reshape(self.nely, self.nelx),
                    target_count=k,
                )
            sample = self._finalize_density_sample(sample_raw)
            if self.config.removal_ladder_connectivity_max is not None:
                connectivities.append(self.disconnected_solid_fraction(sample))
            compliances.append(self._solve_compliance(sample))

        connectivity_max = self.config.removal_ladder_connectivity_max
        if targets:
            values = []
            for idx, (compliance, target) in enumerate(
                zip(compliances, targets, strict=True)
            ):
                if connectivity_max is not None:
                    values.append(max(connectivities[idx] - connectivity_max, 0.0))
                values.append(max(compliance - target, 0.0))
            # Keep the final three columns compatible with the existing
            # volume/roughness/compliance summary machinery.
            values.extend([0.0, 0.0, float(compliances[-1])])
            return values
        values = []
        for idx, compliance in enumerate(compliances):
            if connectivity_max is not None:
                values.append(max(connectivities[idx] - connectivity_max, 0.0))
            values.append(float(compliance))
        values.extend([0.0, 0.0, float(compliances[-1])])
        return values

    def _format_objective_values(
        self,
        volume: float,
        roughness: float,
        connectivity: float,
        diversity: float,
        compliance: float,
    ) -> list[float]:
        if self.config.use_levels_ladder:
            objective_values = {
                "volume": volume,
                "roughness": roughness,
                "connectivity": connectivity,
                "diversity": diversity,
                "compliance": compliance,
            }
            return [
                objective_values[name] for name in self.config.levels_ladder_objectives
            ]

        level_values: list[float] = []
        if self.config.ladder_sequence:
            for kind, bound in self.config.ladder_sequence:
                if kind == "volume":
                    level_values.append(max(volume - bound, 0.0))
                elif kind == "compliance":
                    level_values.append(max(compliance - bound, 0.0))
                elif kind == "roughness":
                    level_values.append(max(roughness - bound, 0.0))
                elif kind == "connectivity":
                    level_values.append(max(connectivity - bound, 0.0))
                elif kind == "diversity":
                    level_values.append(max(bound - diversity, 0.0))
                else:
                    raise ValueError(f"Unknown ladder kind: {kind}")
        else:
            for bound in self.config.volume_ladder:
                level_values.append(max(volume - bound, 0.0))
            for bound in self.config.compliance_ladder:
                level_values.append(max(compliance - bound, 0.0))
            for bound in self.config.roughness_ladder:
                level_values.append(max(roughness - bound, 0.0))
            for bound in self.config.connectivity_ladder:
                level_values.append(max(connectivity - bound, 0.0))
            for bound in self.config.diversity_ladder:
                level_values.append(max(bound - diversity, 0.0))
        if self.config.connectivity_max is not None:
            level_values.append(max(connectivity - self.config.connectivity_max, 0.0))
        level_values.extend(
            [
                max(volume - self.config.volume_max, 0.0),
                max(roughness - self.config.roughness_max, 0.0),
                compliance,
            ]
        )
        return level_values

    def evaluate_densities_numpy(self, x_phys: np.ndarray) -> list[list[float]]:
        samples = np.asarray(x_phys, dtype=np.float64).reshape(-1, self.nely, self.nelx)
        if self.config.compliance_solver == "matrix_free_cg":
            compliances = self._solve_compliance_matrix_free_batch(samples)
            objectives = [
                (
                    float(sample.mean()),
                    0.5
                    * (
                        float(np.abs(sample[:, 1:] - sample[:, :-1]).mean())
                        + float(np.abs(sample[1:, :] - sample[:-1, :]).mean())
                    ),
                    self.disconnected_solid_fraction(sample)
                    if self.requires_connectivity
                    else 0.0,
                    self.boundary_chamfer_diversity(sample)
                    if self.config.diversity_ladder
                    or any(
                        kind == "diversity"
                        for kind, _bound in self.config.ladder_sequence
                    )
                    or (
                        self.config.use_levels_ladder
                        and "diversity" in self.config.levels_ladder_objectives
                    )
                    else float("inf"),
                    float(compliance),
                )
                for sample, compliance in zip(samples, compliances, strict=True)
            ]
        elif self.config.fem_workers <= 1 or len(samples) <= 1:
            objectives = [self.density_objectives(sample) for sample in samples]
        else:
            pool = self._get_fem_executor()
            objectives = list(pool.map(self.density_objectives, samples))
        return [
            self._format_objective_values(
                volume,
                roughness,
                connectivity,
                diversity,
                compliance,
            )
            for volume, roughness, connectivity, diversity, compliance in objectives
        ]

    def evaluate_removal_ladder_numpy(self, x_np: np.ndarray) -> list[list[float]]:
        scores = np.asarray(x_np, dtype=np.float32).reshape(-1, self.nelems)
        if self.config.fem_workers <= 1 or len(scores) <= 1:
            return [self.removal_ladder_objectives(sample) for sample in scores]
        pool = self._get_fem_executor()
        return list(pool.map(self.removal_ladder_objectives, scores))

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[list[float]]:
        if isinstance(theta, torch.Tensor):
            x_np = theta.detach().to("cpu", torch.float32).numpy()
        else:
            x_np = np.asarray(theta, dtype=np.float32)

        if self.config.removal_ladder_volumes:
            return self.evaluate_removal_ladder_numpy(x_np)
        return self.evaluate_densities_numpy(self.decode_designs_numpy(x_np))


class TorchFEMCantileverEvaluator(FEMCantileverEvaluator):
    def __init__(
        self, config: FEMConfig, *, torchfem_src: str | None = None, device: str = "cpu"
    ) -> None:
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        ensure_torchfem_importable(torchfem_src)
        from torchfem import Planar
        from torchfem.materials import IsotropicElasticityPlaneStress
        from torchfem.mesh import rect_quad

        self.config = config
        self.nelx = config.grid_width
        self.nely = config.grid_height
        self._configure_code_shape()
        self.tiny_decoder: Any | None = None
        self.nelems = self.nelx * self.nely
        self.device = torch.device(device)
        self.dtype = torch.float64

        nodes, elements = rect_quad(
            self.nelx + 1, self.nely + 1, float(self.nelx), float(self.nely)
        )
        self.nodes = nodes.to(self.device, self.dtype)
        self.elements = elements.to(self.device)
        self.Planar = Planar
        self.material_class = IsotropicElasticityPlaneStress
        self.filter_offsets, self.filter_weights = self._build_filter_kernel()
        self.sorted_material_values = fixed_sorted_material_values_numpy(
            self.nelems,
            self.config.volume_max,
            profile=self.config.sorted_material_profile,
            steepness=self.config.sorted_material_steepness,
        )
        self.diversity_reference_designs: np.ndarray | None = None
        self.solid_compliance = self._compute_solid_compliance()
        torch.set_default_dtype(prev_dtype)

    def _build_model(self, density_phys: np.ndarray):
        penalized = self.config.e_min + (
            density_phys.T.ravel(order="C") ** self.config.simp_p
        ) * (self.config.e_max - self.config.e_min)
        material = self.material_class(
            E=torch.as_tensor(penalized, device=self.device, dtype=self.dtype),
            nu=torch.as_tensor(
                self.config.poisson_ratio, device=self.device, dtype=self.dtype
            ),
        )
        model = self.Planar(self.nodes, self.elements, material, thickness=1.0)
        model.etype.ipoints = model.etype.ipoints.to(self.device, self.dtype)
        model.etype.iweights = model.etype.iweights.to(self.device, self.dtype)
        left = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
        right = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max())
        model.constraints[left, :] = True
        candidates = torch.where(right)[0]
        target_y = torch.tensor(self.nely / 2.0, device=self.device, dtype=self.dtype)
        tip = candidates[torch.argmin(torch.abs(model.nodes[candidates, 1] - target_y))]
        model.forces[tip, 1] = -self.config.load_scale
        return model

    def _solve_compliance(self, density_phys: np.ndarray) -> float:
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        model = self._build_model(density_phys)
        u, f, _, _, _ = model.solve(method="spsolve")
        torch.set_default_dtype(prev_dtype)
        return float(torch.inner(f.ravel(), u.ravel()).item())


class BufferChamferDiversityObjective:
    """Callable objective that refreshes Chamfer references from the elite buffer."""

    def __init__(self, evaluator: FEMCantileverEvaluator, buffer: Buffer) -> None:
        self.evaluator = evaluator
        self.buffer = buffer

    def _refresh_references(self) -> None:
        cfg = self.evaluator.config
        needs_diversity = (
            bool(cfg.diversity_ladder)
            or any(kind == "diversity" for kind, _bound in cfg.ladder_sequence)
            or (cfg.use_levels_ladder and "diversity" in cfg.levels_ladder_objectives)
        )
        if not needs_diversity or len(self.buffer) == 0:
            self.evaluator.diversity_reference_designs = None
            return
        k = min(cfg.diversity_reference_size, len(self.buffer))
        tensors = self.buffer.get_top_k(k).detach().cpu().numpy()
        self.evaluator.diversity_reference_designs = (
            self.evaluator.decode_designs_numpy(tensors)
        )

    def evaluate_densities_numpy(self, x_phys: np.ndarray) -> list[list[float]]:
        self._refresh_references()
        return self.evaluator.evaluate_densities_numpy(x_phys)

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[list[float]]:
        self._refresh_references()
        return self.evaluator(theta)


def pairwise_l2_mean(designs: torch.Tensor) -> float:
    flat = designs.reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = flat[:, None, :] - flat[None, :, :]
    dists = diffs.pow(2).sum(dim=-1).sqrt()
    triu = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    return float(dists[triu[0], triu[1]].mean().item())


def pairwise_hamming_mean(designs: torch.Tensor, threshold: float = 0.5) -> float:
    flat = (designs >= threshold).reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = (flat[:, None, :] != flat[None, :, :]).float().mean(dim=-1)
    triu = torch.triu_indices(flat.shape[0], flat.shape[0], offset=1)
    return float(diffs[triu[0], triu[1]].mean().item())


class DiverseEliteBuffer:
    """Buffer wrapper that keeps a ranked but Hamming-diverse elite set."""

    def __init__(
        self,
        inner: Buffer,
        *,
        min_hamming: float,
        topk_frac: float,
    ) -> None:
        if min_hamming < 0:
            raise ValueError(f"min_hamming must be non-negative, got {min_hamming}")
        if not 0.0 < topk_frac < 1.0:
            raise ValueError(f"topk_frac must be in (0, 1), got {topk_frac}")
        self.inner = inner
        self.min_hamming = min_hamming
        self.topk_frac = topk_frac
        self.buffer_size = inner.buffer_size
        self.value_levels = inner.value_levels

    def _binary_order_proxy(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        flat = torch.stack([tensor.reshape(-1).to(torch.float32) for tensor in tensors])
        k = int(round(self.topk_frac * flat.shape[1]))
        k = min(max(k, 1), flat.shape[1])
        top_idx = torch.topk(flat, k=k, dim=1).indices
        proxy = torch.zeros_like(flat, dtype=torch.bool)
        proxy.scatter_(1, top_idx, True)
        return proxy

    def _current_entries(self) -> list[tuple[list[float], torch.Tensor]]:
        return [
            ([float(v) for v in value], self.inner.get(idx).detach().clone())
            for idx, value in enumerate(self.inner.get_sorted_values())
        ]

    def _select_diverse_entries(
        self,
        entries: list[tuple[list[float], torch.Tensor]],
    ) -> list[tuple[list[float], torch.Tensor]]:
        entries.sort(key=lambda entry: tuple(entry[0]))
        target_size = min(self.buffer_size, len(entries))
        proxies = self._binary_order_proxy([tensor for _value, tensor in entries])
        selected: list[int] = []
        selected_mask = torch.zeros(len(entries), dtype=torch.bool)
        for idx in range(len(entries)):
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
            else:
                distances = (proxies[selected] != proxies[idx]).float().mean(dim=1)
                if float(distances.min().item()) >= self.min_hamming:
                    selected.append(idx)
                    selected_mask[idx] = True
            if len(selected) >= target_size:
                break

        # Always fill the buffer by rank if the diversity constraint is too strict.
        if len(selected) < target_size:
            for idx in range(len(entries)):
                if not bool(selected_mask[idx]):
                    selected.append(idx)
                if len(selected) >= target_size:
                    break
        return [entries[idx] for idx in selected[:target_size]]

    def insert(self, tensor: torch.Tensor, value: float | list[float]) -> None:
        self.insert_many(tensors=[tensor], values=[value])

    def insert_many(
        self,
        tensors: list[torch.Tensor],
        values: list[float] | list[list[float]],
    ) -> None:
        if self.min_hamming <= 0:
            self.inner.insert_many(tensors=tensors, values=values)
            return
        tensor_list = [tensor.detach().clone() for tensor in tensors]
        normalized_values = self.inner._normalize_many_values(tensor_list, values)
        new_entries = [
            (
                [float(v) for v in value]
                if isinstance(value, list | tuple)
                else [float(value)],
                tensor,
            )
            for tensor, value in zip(tensor_list, normalized_values)
        ]
        entries = self._current_entries() + new_entries
        selected_entries = self._select_diverse_entries(entries)
        self.inner.clear()
        self.inner.insert_many(
            tensors=[tensor for _value, tensor in selected_entries],
            values=[value for value, _tensor in selected_entries],
        )

    def __len__(self) -> int:
        return len(self.inner)

    def len(self) -> int:
        return len(self.inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class NicheEliteBuffer:
    """Split elites across design-space niches while exposing the Buffer API."""

    def __init__(
        self,
        *,
        buffer_size: int,
        value_levels: Levels,
        niche_count: int,
        design_proxy: DesignProxyFn,
        min_hamming: float,
        view_mode: Literal["balanced", "global"] = "balanced",
        cross_niche_min_hamming: float = 0.0,
        cross_niche_reference_top_k: int = 1,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if niche_count < 2:
            raise ValueError(f"niche_count must be >= 2, got {niche_count}")
        if not 0.0 <= min_hamming <= 1.0:
            raise ValueError(f"min_hamming must be in [0, 1], got {min_hamming}")
        if not 0.0 <= cross_niche_min_hamming <= 1.0:
            raise ValueError(
                "cross_niche_min_hamming must be in [0, 1], got "
                f"{cross_niche_min_hamming}"
            )
        if cross_niche_reference_top_k <= 0:
            raise ValueError(
                "cross_niche_reference_top_k must be positive, got "
                f"{cross_niche_reference_top_k}"
            )
        if view_mode not in {"balanced", "global"}:
            raise ValueError(f"Unknown niche buffer view_mode: {view_mode}")
        self.buffer_size = buffer_size
        self.value_levels = value_levels
        self.niche_count = niche_count
        self.design_proxy = design_proxy
        self.min_hamming = min_hamming
        self.view_mode = view_mode
        self.cross_niche_min_hamming = cross_niche_min_hamming
        self.cross_niche_reference_top_k = cross_niche_reference_top_k
        base_capacity = buffer_size // niche_count
        remainder = buffer_size % niche_count
        self.niche_capacities = [
            base_capacity + (1 if idx < remainder else 0) for idx in range(niche_count)
        ]
        if min(self.niche_capacities) <= 0:
            raise ValueError(
                "niche_count must not exceed buffer_size, got "
                f"{niche_count} niches for buffer_size={buffer_size}"
            )
        self.buffers = [
            Buffer(buffer_size=capacity, value_levels=value_levels)
            for capacity in self.niche_capacities
        ]
        self.niche_anchor_proxies: list[torch.Tensor | None] = [
            None for _ in range(niche_count)
        ]
        self._pending_initial_niche_labels: list[int] = []

    def queue_initial_niche_labels(self, labels: np.ndarray | list[int]) -> None:
        """Use fixed niche labels for the next insertions, typically init seeds."""
        label_list = [int(label) for label in np.asarray(labels).reshape(-1)]
        for label in label_list:
            if label < 0 or label >= self.niche_count:
                raise ValueError(
                    f"initial niche label must be in [0, {self.niche_count}), got {label}"
                )
        self._pending_initial_niche_labels = label_list

    def _consume_initial_niche_labels(self, count: int) -> list[int] | None:
        if not self._pending_initial_niche_labels:
            return None
        if len(self._pending_initial_niche_labels) < count:
            raise ValueError(
                "Not enough queued initial niche labels for insertion: "
                f"{len(self._pending_initial_niche_labels)} < {count}"
            )
        labels = self._pending_initial_niche_labels[:count]
        self._pending_initial_niche_labels = self._pending_initial_niche_labels[count:]
        return labels

    def _normalize_many_values(
        self,
        tensors: list[torch.Tensor],
        values: list[float] | list[list[float]],
    ) -> list[float | list[float] | tuple[float, ...]]:
        return self.buffers[0]._normalize_many_values(tensors, values)

    @staticmethod
    def _value_row(value: float | list[float] | tuple[float, ...]) -> list[float]:
        if isinstance(value, list | tuple):
            return [float(v) for v in value]
        return [float(value)]

    def _proxy(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.empty((0, 0), dtype=torch.bool)
        batch = torch.stack([tensor.detach().cpu() for tensor in tensors])
        proxy = self.design_proxy(batch)
        if proxy.shape[0] != len(tensors):
            raise ValueError(
                "niche design_proxy must return one proxy per tensor, got "
                f"{proxy.shape[0]} for {len(tensors)}"
            )
        return proxy.reshape(proxy.shape[0], -1).to(torch.bool).cpu()

    def _current_entries(self) -> list[tuple[list[float], torch.Tensor]]:
        entries: list[tuple[list[float], torch.Tensor]] = []
        for buffer in self.buffers:
            entries.extend(
                ([float(v) for v in value], buffer.get(idx).detach().clone())
                for idx, value in enumerate(buffer.get_sorted_values())
            )
        return entries

    @staticmethod
    def _hamming_distance(
        proxy: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        return (references != proxy).to(torch.float32).mean(dim=1)

    def _set_anchor_if_missing(
        self,
        niche_idx: int,
        proxy: torch.Tensor,
    ) -> None:
        if self.niche_anchor_proxies[niche_idx] is None:
            self.niche_anchor_proxies[niche_idx] = proxy.detach().clone().to(torch.bool)

    def _anchor_tensor(self) -> tuple[list[int], torch.Tensor]:
        indices: list[int] = []
        anchors: list[torch.Tensor] = []
        for idx, proxy in enumerate(self.niche_anchor_proxies):
            if proxy is None:
                continue
            indices.append(idx)
            anchors.append(proxy)
        if not anchors:
            return [], torch.empty((0, 0), dtype=torch.bool)
        return indices, torch.stack(anchors)

    def _ensure_anchors_from_candidates(
        self,
        entries: list[tuple[list[float], torch.Tensor]],
        proxies: torch.Tensor,
    ) -> None:
        """Seed missing niche anchors from ranked candidates without moving entries."""
        if proxies.shape[0] == 0:
            return
        ranked_positions = sorted(
            range(len(entries)), key=lambda idx: tuple(entries[idx][0])
        )
        used_positions: set[int] = set()
        for niche_idx in range(self.niche_count):
            if self.niche_anchor_proxies[niche_idx] is not None:
                continue

            _anchor_indices, anchors = self._anchor_tensor()
            if anchors.numel() == 0:
                chosen = ranked_positions[0]
            else:
                candidate_scores: list[tuple[float, int]] = []
                for pos in ranked_positions:
                    if pos in used_positions:
                        continue
                    distance = self._hamming_distance(proxies[pos], anchors).min()
                    candidate_scores.append((float(distance.item()), pos))
                if not candidate_scores:
                    break
                _distance, chosen = max(candidate_scores, key=lambda item: item[0])

            used_positions.add(chosen)
            self._set_anchor_if_missing(niche_idx, proxies[chosen])

    def _assign_niche_indices(
        self,
        entries: list[tuple[list[float], torch.Tensor]],
    ) -> list[int]:
        proxies = self._proxy([tensor for _value, tensor in entries])
        self._ensure_anchors_from_candidates(entries, proxies)
        anchor_indices, anchors = self._anchor_tensor()
        if anchors.numel() == 0:
            return [0 for _ in entries]

        labels: list[int] = []
        for proxy in proxies:
            distances = self._hamming_distance(proxy, anchors)
            nearest_anchor = int(torch.argmin(distances).item())
            labels.append(anchor_indices[nearest_anchor])
        return labels

    def _select_representatives(self, proxies: torch.Tensor) -> list[int]:
        selected: list[int] = []
        selected_mask = torch.zeros(proxies.shape[0], dtype=torch.bool)
        for idx in range(proxies.shape[0]):
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
            distances = self._hamming_distance(proxies[idx], proxies[selected])
            if float(distances.min().item()) >= self.min_hamming:
                selected.append(idx)
                selected_mask[idx] = True
            if len(selected) >= self.niche_count:
                return selected

        while len(selected) < min(self.niche_count, proxies.shape[0]):
            if not selected:
                break
            distances_to_selected = torch.stack(
                [
                    self._hamming_distance(proxies[idx], proxies[selected]).min()
                    for idx in range(proxies.shape[0])
                ]
            )
            distances_to_selected[selected_mask] = -1.0
            next_idx = int(torch.argmax(distances_to_selected).item())
            if bool(selected_mask[next_idx]):
                break
            selected.append(next_idx)
            selected_mask[next_idx] = True
        return selected

    def _rebuild(self, entries: list[tuple[list[float], torch.Tensor]]) -> None:
        entries.sort(key=lambda entry: tuple(entry[0]))
        if not entries:
            for buffer in self.buffers:
                buffer.clear()
            return
        proxies = self._proxy([tensor for _value, tensor in entries])
        representatives = self._select_representatives(proxies)
        niche_indices: list[list[int]] = [[] for _ in range(self.niche_count)]
        if not representatives:
            representatives = [0]
        rep_proxies = proxies[representatives]
        for idx in range(len(entries)):
            distances = self._hamming_distance(proxies[idx], rep_proxies)
            niche_idx = int(torch.argmin(distances).item())
            niche_indices[niche_idx].append(idx)

        selected_by_niche: list[list[int]] = []
        selected: set[int] = set()
        for capacity, indices in zip(self.niche_capacities, niche_indices, strict=True):
            local_selected = indices[:capacity]
            selected_by_niche.append(local_selected)
            selected.update(local_selected)

        # If nearest-representative assignment is imbalanced, fill underfull niches
        # from ranked overflow rather than silently shrinking the total buffer.
        unused = [idx for idx in range(len(entries)) if idx not in selected]
        unused_position = 0
        for niche_idx, capacity in enumerate(self.niche_capacities):
            need = capacity - len(selected_by_niche[niche_idx])
            if need <= 0:
                continue
            fill = unused[unused_position : unused_position + need]
            unused_position += len(fill)
            selected_by_niche[niche_idx].extend(fill)

        for buffer, capacity, indices_for_niche in zip(
            self.buffers,
            self.niche_capacities,
            selected_by_niche,
            strict=True,
        ):
            buffer.clear()
            indices_for_niche = indices_for_niche[:capacity]
            if indices_for_niche:
                selected_entries = [entries[idx] for idx in indices_for_niche]
                buffer.insert_many(
                    tensors=[tensor for _value, tensor in selected_entries],
                    values=[value for value, _tensor in selected_entries],
                )

    def reset_anchors_from_current_tops(self) -> None:
        """Refresh niche anchors from the current best entry in each niche."""
        self.niche_anchor_proxies = [None for _ in range(self.niche_count)]
        for niche_idx, buffer in enumerate(self.buffers):
            if len(buffer) == 0:
                continue
            top = buffer.get_top_k(1)
            tensor = top[0] if top.ndim > 1 else top
            proxy = self._proxy([tensor])[0]
            self._set_anchor_if_missing(niche_idx, proxy)

    def rebuild_from_entries(
        self,
        entries: list[tuple[list[float], torch.Tensor]],
    ) -> None:
        """Rebuild all niche sub-buffers from already-evaluated entries."""
        self._rebuild(entries)
        self.reset_anchors_from_current_tops()

    def _passes_cross_niche_distance(
        self,
        *,
        niche_idx: int,
        proxy: torch.Tensor,
    ) -> bool:
        if self.cross_niche_min_hamming <= 0:
            return True

        references: list[torch.Tensor] = []
        for other_idx, buffer in enumerate(self.buffers):
            if other_idx == niche_idx or len(buffer) == 0:
                continue
            k = min(self.cross_niche_reference_top_k, len(buffer))
            top = buffer.get_top_k(k)
            if top.ndim == 1:
                references.append(top.detach().cpu())
            else:
                references.extend(row.detach().cpu() for row in top)
        if not references:
            return True

        reference_proxies = self._proxy(references)
        min_distance = self._hamming_distance(proxy, reference_proxies).min()
        return float(min_distance.item()) >= self.cross_niche_min_hamming

    def insert(self, tensor: torch.Tensor, value: float | list[float]) -> None:
        self.insert_many(tensors=[tensor], values=[value])

    def insert_many(
        self,
        tensors: list[torch.Tensor],
        values: list[float] | list[list[float]],
    ) -> None:
        tensor_list = [tensor.detach().clone() for tensor in tensors]
        normalized_values = self._normalize_many_values(tensor_list, values)
        if len(normalized_values) != len(tensor_list):
            raise ValueError(
                f"Number of values ({len(normalized_values)}) does not match "
                f"number of tensors ({len(tensor_list)})"
            )
        initial_labels = self._consume_initial_niche_labels(len(tensor_list))
        if initial_labels is not None:
            proxies = self._proxy(tensor_list)
            for niche_idx in range(self.niche_count):
                selected = []
                for tensor, value, proxy, label in zip(
                    tensor_list,
                    normalized_values,
                    proxies,
                    initial_labels,
                    strict=True,
                ):
                    if label != niche_idx:
                        continue
                    self._set_anchor_if_missing(niche_idx, proxy)
                    selected.append((tensor, value))
                if selected:
                    self.buffers[niche_idx].insert_many(
                        tensors=[tensor for tensor, _value in selected],
                        values=[value for _tensor, value in selected],
                    )
            return
        new_entries = [
            (self._value_row(value), tensor)
            for tensor, value in zip(tensor_list, normalized_values, strict=True)
        ]
        niche_labels = self._assign_niche_indices(new_entries)
        proxies = (
            self._proxy([tensor for _value, tensor in new_entries])
            if self.cross_niche_min_hamming > 0
            else None
        )
        for niche_idx in range(self.niche_count):
            selected = [
                (tensor, value)
                for position, ((value, tensor), label) in enumerate(
                    zip(new_entries, niche_labels, strict=True)
                )
                if label == niche_idx
                and (
                    proxies is None
                    or self._passes_cross_niche_distance(
                        niche_idx=niche_idx,
                        proxy=proxies[position],
                    )
                )
            ]
            if selected:
                self.buffers[niche_idx].insert_many(
                    tensors=[tensor for tensor, _value in selected],
                    values=[value for _tensor, value in selected],
                )

    def _global_entries(self) -> list[tuple[list[float], torch.Tensor]]:
        entries = self._current_entries()
        entries.sort(key=lambda entry: tuple(entry[0]))
        return entries[: self.buffer_size]

    def _balanced_entries(
        self,
        limit: int | None = None,
    ) -> list[tuple[list[float], torch.Tensor]]:
        per_niche_entries = []
        for buffer in self.buffers:
            per_niche_entries.append(
                [
                    ([float(v) for v in value], buffer.get(idx).detach().clone())
                    for idx, value in enumerate(buffer.get_sorted_values())
                ]
            )
        limit = self.buffer_size if limit is None else min(limit, self.buffer_size)
        selected: list[tuple[list[float], torch.Tensor]] = []
        local_idx = 0
        while len(selected) < limit:
            added = False
            for entries in per_niche_entries:
                if local_idx < len(entries):
                    selected.append(entries[local_idx])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            local_idx += 1
        selected.sort(key=lambda entry: tuple(entry[0]))
        return selected

    def _view_entries(
        self,
        limit: int | None = None,
    ) -> list[tuple[list[float], torch.Tensor]]:
        if self.view_mode == "global":
            entries = self._global_entries()
            if limit is not None:
                entries = entries[:limit]
            return entries
        return self._balanced_entries(limit=limit)

    def get_sorted_values(self) -> list[list[float]]:
        return [value for value, _tensor in self._view_entries()]

    def get_niche_sorted_values(self, niche_idx: int) -> list[list[float]]:
        if niche_idx < 0 or niche_idx >= self.niche_count:
            raise IndexError(
                f"niche_idx must be in [0, {self.niche_count}), got {niche_idx}"
            )
        return [
            [float(v) for v in value]
            for value in self.buffers[niche_idx].get_sorted_values()
        ]

    def get_niche_top_k(self, niche_idx: int, k: int) -> torch.Tensor:
        if niche_idx < 0 or niche_idx >= self.niche_count:
            raise IndexError(
                f"niche_idx must be in [0, {self.niche_count}), got {niche_idx}"
            )
        k = min(max(k, 0), len(self.buffers[niche_idx]))
        return self.buffers[niche_idx].get_top_k(k)

    def get_niche_length(self, niche_idx: int) -> int:
        if niche_idx < 0 or niche_idx >= self.niche_count:
            raise IndexError(
                f"niche_idx must be in [0, {self.niche_count}), got {niche_idx}"
            )
        return len(self.buffers[niche_idx])

    def get_niche(self, niche_idx: int, idx: int) -> torch.Tensor:
        if niche_idx < 0 or niche_idx >= self.niche_count:
            raise IndexError(
                f"niche_idx must be in [0, {self.niche_count}), got {niche_idx}"
            )
        return self.buffers[niche_idx].get(idx)

    def get(self, idx: int | slice) -> torch.Tensor:
        entries = self._view_entries()
        if not entries:
            raise RuntimeError("Buffer is empty")
        if isinstance(idx, int):
            idx %= len(entries)
            return entries[idx][1]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(entries))
            tensors = [entries[pos][1] for pos in range(start, stop, step)]
            if not tensors:
                return entries[0][1].unsqueeze(0)[:0]
            return torch.stack(tensors)
        raise TypeError("Index must be int or slice")

    def __getitem__(self, idx: int | slice) -> torch.Tensor:
        return self.get(idx)

    def get_top_k(self, k: int) -> torch.Tensor:
        k = min(max(k, 0), len(self))
        entries = self._view_entries(limit=k)
        if not entries:
            first_tensor = next(
                (
                    buffer.tensor_buffer
                    for buffer in self.buffers
                    if buffer.tensor_buffer is not None
                ),
                None,
            )
            if first_tensor is None:
                raise RuntimeError("Buffer is empty")
            return first_tensor[:0]
        return torch.stack([tensor for _value, tensor in entries])

    def get_bottom_k(self, k: int) -> torch.Tensor:
        entries = self._view_entries()
        k = min(max(k, 0), len(entries))
        if k == 0:
            return self.get_top_k(0)
        return torch.stack([tensor for _value, tensor in entries[-k:]])

    def get_top_p(self, p: float) -> torch.Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_top_k(int(p * len(self)))

    def get_bottom_p(self, p: float) -> torch.Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_bottom_k(int(p * len(self)))

    @staticmethod
    def _stack_sampled_entries(
        entries: list[tuple[list[float], torch.Tensor]],
        positions: list[int],
    ) -> torch.Tensor:
        return torch.stack([entries[pos][1] for pos in positions])

    def get_random_batch(self, batch_size: int) -> torch.Tensor:
        if batch_size == 0:
            return self.get_top_k(0)
        entries = self._view_entries()
        if batch_size > len(entries):
            raise ValueError(
                f"batch_size {batch_size} exceeds current buffer length {len(entries)}"
            )
        positions = random.sample(range(len(entries)), batch_size)
        return self._stack_sampled_entries(entries, positions)

    def get_random_batch_from_top_p(self, p: float, batch_size: int) -> torch.Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_random_batch_from_top_k(int(p * len(self)), batch_size)

    def get_random_batch_from_top_k(self, k: int, batch_size: int) -> torch.Tensor:
        if batch_size == 0:
            return self.get_top_k(0)
        entries = self._view_entries(limit=min(max(k, 0), len(self)))
        if len(entries) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from top_k={len(entries)}"
            )
        positions = random.sample(range(len(entries)), batch_size)
        return self._stack_sampled_entries(entries, positions)

    def get_random_batch_from_bottom_p(self, p: float, batch_size: int) -> torch.Tensor:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        return self.get_random_batch_from_bottom_k(int(p * len(self)), batch_size)

    def get_random_batch_from_bottom_k(self, k: int, batch_size: int) -> torch.Tensor:
        if batch_size == 0:
            return self.get_top_k(0)
        entries = self._view_entries()
        bottom_entries = entries[-min(max(k, 0), len(entries)) :]
        if len(bottom_entries) < batch_size:
            raise ValueError(
                "Cannot sample "
                f"batch_size={batch_size} from bottom_k={len(bottom_entries)}"
            )
        positions = random.sample(range(len(bottom_entries)), batch_size)
        return self._stack_sampled_entries(bottom_entries, positions)

    def get_value(self, index: int, level: int = 0) -> float:
        entries = self._view_entries()
        if not entries:
            raise IndexError("Buffer is empty")
        index %= len(entries)
        level %= self.value_levels.num_levels()
        return entries[index][0][level]

    def get_mean_buffer_value(self, level: int = 0) -> float:
        values = self.get_sorted_values()
        if not values:
            raise IndexError("Buffer is empty")
        if level < 0:
            level = self.value_levels.num_levels() + level
        return float(np.mean([value[level] for value in values]))

    def len(self) -> int:
        return min(self.buffer_size, sum(len(buffer) for buffer in self.buffers))

    def __len__(self) -> int:
        return self.len()

    def clear(self) -> None:
        for buffer in self.buffers:
            buffer.clear()
        self.niche_anchor_proxies = [None for _ in range(self.niche_count)]

    def get_niche_stats(self) -> list[dict[str, float]]:
        """Return per-niche telemetry for experiment history."""
        top_tensors: list[torch.Tensor | None] = []
        stats: list[dict[str, float]] = []
        for niche_idx, (capacity, buffer) in enumerate(
            zip(self.niche_capacities, self.buffers, strict=True)
        ):
            if len(buffer) == 0:
                top_tensors.append(None)
                stats.append(
                    {
                        "niche_index": float(niche_idx),
                        "capacity": float(capacity),
                        "size": 0.0,
                        "best_last": float("nan"),
                        "best_feasible_last": float("nan"),
                        "mean_last": float("nan"),
                        "median_last": float("nan"),
                        "p10_last": float("nan"),
                        "p90_last": float("nan"),
                        "representative_min_hamming": float("nan"),
                        "representative_mean_hamming": float("nan"),
                    }
                )
                continue

            values = np.asarray(buffer.get_sorted_values(), dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            last = values[:, -1]
            best_feasible_last = float("nan")
            if values.shape[1] >= 3:
                feasible = (values[:, -3] <= 1e-6) & (values[:, -2] <= 1e-6)
                if np.any(feasible):
                    best_feasible_last = float(np.min(last[feasible]))
            top_tensors.append(buffer.get_top_k(1)[0].detach().cpu())
            stats.append(
                {
                    "niche_index": float(niche_idx),
                    "capacity": float(capacity),
                    "size": float(len(buffer)),
                    "best_last": float(last[0]),
                    "best_feasible_last": best_feasible_last,
                    "mean_last": float(np.mean(last)),
                    "median_last": float(np.median(last)),
                    "p10_last": float(np.percentile(last, 10)),
                    "p90_last": float(np.percentile(last, 90)),
                    "representative_min_hamming": float("nan"),
                    "representative_mean_hamming": float("nan"),
                }
            )

        valid_tensors = [tensor for tensor in top_tensors if tensor is not None]
        if len(valid_tensors) >= 2:
            proxies = self._proxy(valid_tensors)
            valid_position = 0
            for idx, tensor in enumerate(top_tensors):
                if tensor is None:
                    continue
                distances = self._hamming_distance(proxies[valid_position], proxies)
                distances = torch.cat(
                    [
                        distances[:valid_position],
                        distances[valid_position + 1 :],
                    ]
                )
                stats[idx]["representative_min_hamming"] = float(distances.min().item())
                stats[idx]["representative_mean_hamming"] = float(
                    distances.mean().item()
                )
                valid_position += 1
        return stats


def objective_value_row(value: Any) -> list[float]:
    """Normalize one objective result row to a list of floats."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return [float(array)]
    return [float(item) for item in array.reshape(-1)]


def buffer_current_entries(buffer_impl: Any) -> list[tuple[list[float], torch.Tensor]]:
    """Return ranked buffer entries without depending on a specific wrapper."""
    if hasattr(buffer_impl, "_current_entries"):
        return buffer_impl._current_entries()
    return [
        ([float(v) for v in value], buffer_impl.get(idx).detach().clone())
        for idx, value in enumerate(buffer_impl.get_sorted_values())
    ]


def rebuild_buffer_entries(
    buffer_impl: Any,
    entries: list[tuple[list[float], torch.Tensor]],
) -> None:
    """Clear and repopulate a buffer wrapper from evaluated entries."""
    normalized_entries = [
        ([float(v) for v in value], tensor.detach().clone())
        for value, tensor in entries
    ]
    if isinstance(buffer_impl, NicheEliteBuffer):
        buffer_impl.rebuild_from_entries(normalized_entries)
        return

    tensors = [tensor for _value, tensor in normalized_entries]
    values = [value for value, _tensor in normalized_entries]
    if isinstance(buffer_impl, DiverseEliteBuffer):
        buffer_impl.inner.clear()
        if tensors:
            buffer_impl.insert_many(tensors=tensors, values=values)
        return

    buffer_impl.clear()
    if tensors:
        buffer_impl.insert_many(tensors=tensors, values=values)


def reevaluate_buffer_entries(
    buffer_impl: Any,
    objective_fn: Callable[[torch.Tensor], list[list[float]]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> int:
    """Re-score all current buffer entries under the current objective."""
    entries = buffer_current_entries(buffer_impl)
    if not entries:
        return 0
    chunk_size = max(int(batch_size), 1)
    revalued: list[tuple[list[float], torch.Tensor]] = []
    for start in range(0, len(entries), chunk_size):
        chunk = entries[start : start + chunk_size]
        batch = torch.stack([tensor for _value, tensor in chunk]).to(
            device=device,
            dtype=dtype,
        )
        values = objective_fn(batch)
        if len(values) != len(chunk):
            raise ValueError(
                "Objective returned "
                f"{len(values)} rows for re-evaluation batch of size {len(chunk)}"
            )
        revalued.extend(
            (objective_value_row(value), tensor)
            for value, (_old_value, tensor) in zip(values, chunk, strict=True)
        )
    rebuild_buffer_entries(buffer_impl, revalued)
    return len(revalued)


def save_design_grid(
    designs: torch.Tensor,
    actual_compliance: np.ndarray,
    relative_compliance: np.ndarray,
    volume_violation: np.ndarray,
    roughness_violation: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_designs = designs.shape[0]
    cols = min(3, num_designs)
    rows = int(np.ceil(num_designs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= num_designs:
            ax.axis("off")
            continue
        ax.imshow(designs[idx].cpu().numpy(), cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            (
                f"#{idx + 1}\nvol={volume_violation[idx]:.3f} rough={roughness_violation[idx]:.3f} "
                f"comp={actual_compliance[idx]:.3f}\nrel={relative_compliance[idx]:.3f}"
            ),
            fontsize=9,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_per_niche_design_grid(
    designs: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a grid of top designs grouped by niche."""
    if designs.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    niche_count, top_k = designs.shape[:2]
    fig, axes = plt.subplots(
        niche_count,
        top_k,
        figsize=(3.6 * top_k, 2.4 * niche_count),
        squeeze=False,
    )
    for niche_idx in range(niche_count):
        for rank_idx in range(top_k):
            ax = axes[niche_idx, rank_idx]
            design = designs[niche_idx, rank_idx]
            if not np.isfinite(design).any():
                ax.axis("off")
                continue
            compliance = float(values[niche_idx, rank_idx, -1])
            ax.imshow(design, cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"niche {niche_idx}, rank {rank_idx + 1}\ncomp={compliance:.3f}",
                fontsize=8,
            )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_best_design_history_grid(
    design_history: np.ndarray,
    value_history: np.ndarray,
    iterations: np.ndarray,
    output_path: Path,
    *,
    title: str,
    max_frames: int = 24,
) -> None:
    """Save a compact visual timeline of the best decoded design."""
    if design_history.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(design_history.shape[0])
    if n_frames > max_frames:
        frame_idx = np.linspace(0, n_frames - 1, max_frames, dtype=np.int32)
    else:
        frame_idx = np.arange(n_frames, dtype=np.int32)

    cols = min(6, len(frame_idx))
    rows = int(np.ceil(len(frame_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for plot_idx in range(rows * cols):
        ax = axes[plot_idx // cols, plot_idx % cols]
        if plot_idx >= len(frame_idx):
            ax.axis("off")
            continue
        history_idx = int(frame_idx[plot_idx])
        design = design_history[history_idx, 0]
        values = value_history[history_idx, 0]
        compliance = float(values[-1]) if values.size else float("nan")
        ax.imshow(design, cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"iter={int(iterations[history_idx])}\ncomp={compliance:.3f}",
            fontsize=8,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


class DecodedDensityOptMixin:
    """Optimizer mixin that trains GAN losses on decoded physical densities."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        decoder: Callable[[torch.Tensor], torch.Tensor],
        density_objective: Callable[[np.ndarray], list[list[float]]],
    ) -> None:
        self.decoder = decoder
        self.density_objective = density_objective
        super().__init__(opt_components)

    def init_buffer(self) -> None:
        n_iter = math.ceil(self.buffer.B.buffer_size / self.components.batch_size)
        logger.info(
            f"Filling decoded-density buffer of size {self.buffer.B.buffer_size} with {n_iter} iterations"
        )
        for _ in range(n_iter):
            with torch.no_grad():
                proposals = self._sample_generator_output()
            self.evaluate(proposals)

    def _sample_generator_output(self) -> torch.Tensor:
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        raw = self.gan.G(z)
        decoded = self.decoder(raw)
        return decoded.reshape(decoded.shape[0], -1)

    def evaluate(self, proposals: torch.Tensor) -> None:
        densities = proposals.detach()
        values = self.density_objective(densities.to("cpu", torch.float32).numpy())
        self.buffer.B.insert_many(values=values, tensors=list(densities))


class DecodedDensityDefaultOpt(DecodedDensityOptMixin, DefaultOpt):
    pass


class DecodedDensityHingeGANOpt(DecodedDensityOptMixin, HingeGANOpt):
    pass


class DecodedDensityLSGANOpt(DecodedDensityOptMixin, LSGANOpt):
    pass


class DecodedDensityWGANOpt(DecodedDensityOptMixin, WGANOpt):
    pass


class DecodedDensityWGANGPOpt(DecodedDensityOptMixin, WGANGPOpt):
    pass


class TopologySpaceUniformity(torch.nn.Module):
    """Wang-Isola uniformity on decoded topology fields instead of raw codes."""

    def __init__(
        self,
        *,
        evaluator: FEMCantileverEvaluator,
        buffer: Buffer | None,
        weight: float,
        t: float = 2.0,
        use_buffer: bool = True,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.buffer = buffer
        self.weight = weight
        self.t = t
        self.use_buffer = use_buffer
        self.scheduler = scheduler

    def _decode_topology_proxy(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.evaluator.config
        needs_postprocess = True
        if cfg.encoding == "topk_volume":
            scores = x.reshape(-1, self.evaluator.nely, self.evaluator.nelx)
            k = int(round(cfg.volume_max * self.evaluator.nely * self.evaluator.nelx))
            hard = project_scores_to_topk_binary_torch(scores, k)
            soft = sigmoid_with_target_mean_torch(scores, target_mean=cfg.volume_max)
            density = hard + soft - soft.detach()
        elif cfg.encoding == "coarse_topk_volume":
            coarse_scores = x.reshape(
                -1, 1, self.evaluator.code_height, self.evaluator.code_width
            )
            upsampled = F.interpolate(
                coarse_scores,
                size=(self.evaluator.nely, self.evaluator.nelx),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            k = int(round(cfg.volume_max * self.evaluator.nely * self.evaluator.nelx))
            hard = project_scores_to_topk_binary_torch(upsampled, k)
            soft = sigmoid_with_target_mean_torch(upsampled, target_mean=cfg.volume_max)
            density = hard + soft - soft.detach()
        elif cfg.encoding in {"direct", "soft_volume", "coarse", "coarse_residual"}:
            density = self.evaluator.decode_designs_torch(x)
            needs_postprocess = False
        else:
            raise ValueError(
                f"Topology-space curiosity does not support encoding={cfg.encoding}"
            )

        if needs_postprocess:
            density = self.evaluator._apply_density_filter_torch(density)
            density = apply_projection_torch(
                density,
                beta=cfg.projection_beta,
                eta=cfg.projection_eta,
                hard_binarize=False,
            )
        return density.reshape(density.shape[0], -1)

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        decoded = self._decode_topology_proxy(g_out)
        if self.use_buffer and self.buffer is not None and len(self.buffer) > 0:
            k = min(g_out.size(0), len(self.buffer))
            buffer_raw = self.buffer.get_top_k(k).to(
                device=g_out.device, dtype=g_out.dtype
            )
            decoded_buffer = self._decode_topology_proxy(buffer_raw).detach()
            decoded = torch.cat([decoded, decoded_buffer], dim=0)
        sched_value = self.scheduler.step() if self.scheduler else 1.0
        return sched_value * self.weight * uniformity_loss(decoded, t=self.t)


class PlummerEmbeddingRepulsion(torch.nn.Module):
    """Differentiable inverse-power repulsion on generated genome vectors."""

    def __init__(
        self,
        *,
        buffer: Any | None,
        weight: float,
        power: float = 1.0,
        eps: float = 1e-3,
        normalize: str = "layernorm",
        terms: str = "batch_buffer",
        use_buffer: bool = True,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        if power <= 0:
            raise ValueError(f"plummer power must be positive, got {power}")
        if eps <= 0:
            raise ValueError(f"plummer eps must be positive, got {eps}")
        if normalize not in {"none", "layernorm", "l2"}:
            raise ValueError(
                f"plummer normalize must be one of none, layernorm, l2; got {normalize}"
            )
        if terms not in {"batch", "buffer", "batch_buffer"}:
            raise ValueError(
                f"plummer terms must be one of batch, buffer, batch_buffer; got {terms}"
            )
        self.buffer = buffer
        self.weight = weight
        self.power = power
        self.eps = eps
        self.normalize = normalize
        self.terms = terms
        self.use_buffer = use_buffer
        self.scheduler = scheduler

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        if self.normalize == "none":
            return flat
        if self.normalize == "l2":
            return F.normalize(flat, p=2, dim=1, eps=self.eps)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (flat - mean) / std

    def _repulsion(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        exclude_diagonal: bool,
    ) -> torch.Tensor:
        if a.numel() == 0 or b.numel() == 0:
            return torch.zeros((), device=a.device, dtype=a.dtype)
        if exclude_diagonal and a.shape[0] < 2:
            return torch.zeros((), device=a.device, dtype=a.dtype)

        dist2 = torch.cdist(a, b, p=2).pow(2) / float(a.shape[1])
        kernel = (self.eps + dist2).pow(-0.5 * self.power)
        if exclude_diagonal:
            keep = ~torch.eye(a.shape[0], device=a.device, dtype=torch.bool)
            kernel = kernel[keep]
        return kernel.mean()

    def forward(self, g_out: torch.Tensor) -> torch.Tensor:
        generated = self._embed(g_out)
        loss = torch.zeros((), device=g_out.device, dtype=g_out.dtype)
        if self.terms in {"batch", "batch_buffer"}:
            loss = loss + self._repulsion(
                generated,
                generated,
                exclude_diagonal=True,
            )

        if (
            self.terms in {"buffer", "batch_buffer"}
            and self.use_buffer
            and self.buffer is not None
            and len(self.buffer) > 0
        ):
            k = min(g_out.shape[0], len(self.buffer))
            elite_raw = self.buffer.get_top_k(k).to(
                device=g_out.device,
                dtype=g_out.dtype,
            )
            elite = self._embed(elite_raw).detach()
            loss = loss + self._repulsion(
                generated,
                elite,
                exclude_diagonal=False,
            )

        sched_value = self.scheduler.step() if self.scheduler else 1.0
        return sched_value * self.weight * loss


def lexicographic_order(values: list[list[float]]) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: tuple(values[idx]))


def plackett_luce_loss(scores_best_to_worst: torch.Tensor) -> torch.Tensor:
    scores = scores_best_to_worst.reshape(-1)
    if scores.numel() < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    log_denoms = torch.logcumsumexp(scores.flip(0), dim=0).flip(0)
    return -(scores - log_denoms).mean()


def contextual_plackett_luce_generator_loss(
    proposal_scores: torch.Tensor,
    context_scores: torch.Tensor,
) -> torch.Tensor:
    """Loss for making each proposal rank above an evaluated context list."""
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
    curve: str = "linear",
    tau: float = 16.0,
) -> torch.Tensor:
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


def utility_targets_from_values(
    values: list[list[float]] | np.ndarray,
    *,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map evaluated objective values to calibrated utilities."""
    if scale <= 0:
        raise ValueError(f"utility scale must be positive, got {scale}")
    values_array = np.asarray(values, dtype=np.float32)
    if values_array.ndim == 1:
        values_array = values_array[:, None]
    utilities = -values_array[:, -1] / float(scale)
    return torch.as_tensor(utilities, device=device, dtype=dtype)


def log_compliance_utility_targets(
    compliance: torch.Tensor,
    *,
    reference: float,
    clip: float,
) -> torch.Tensor:
    """Map compliance to bounded utility; higher is better."""
    if reference <= 0:
        raise ValueError(f"utility_target_scale must be positive, got {reference}")
    targets = -torch.log(torch.clamp(compliance / reference, min=1e-8))
    if clip > 0:
        targets = torch.clamp(targets, min=-clip, max=clip)
    return targets


def split_rank_value_scores(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a two-head discriminator output into rank and value scores."""
    if scores.ndim == 1:
        raise ValueError(
            "Value-augmented ranker requires discriminator output_dim >= 2, "
            f"got 1D output with shape {tuple(scores.shape)}"
        )
    if scores.shape[-1] < 2:
        raise ValueError(
            "Value-augmented ranker requires discriminator output_dim >= 2, "
            f"got shape {tuple(scores.shape)}"
        )
    return scores[..., :1], scores[..., 1:2]


class EvaluatedArchive:
    """Replay archive containing all evaluated tensors and objective values."""

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self.tensors: list[torch.Tensor] = []
        self.values: list[list[float]] = []

    def add_many(self, tensors: torch.Tensor, values: list[list[float]]) -> None:
        for tensor, value in zip(tensors.detach().cpu(), values, strict=True):
            self.tensors.append(tensor.clone())
            self.values.append([float(v) for v in value])
        if self.max_size is not None and len(self.tensors) > self.max_size:
            excess = len(self.tensors) - self.max_size
            del self.tensors[:excess]
            del self.values[:excess]

    def __len__(self) -> int:
        return len(self.tensors)

    def sample_ranked(
        self,
        k: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(self.tensors) == 0:
            raise RuntimeError("Cannot sample from an empty evaluated archive")
        k = min(k, len(self.tensors))
        idx = torch.randperm(len(self.tensors))[:k].tolist()
        ranked_idx = sorted(idx, key=lambda i: tuple(self.values[i]))
        return torch.stack([self.tensors[i] for i in ranked_idx]).to(device, dtype)


class GenomeArchive:
    """Replay archive containing generated genome vectors and objective values."""

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self.genomes: list[torch.Tensor] = []
        self.values: list[list[float]] = []

    def add_many(self, genomes: torch.Tensor, values: list[list[float]]) -> None:
        for genome, value in zip(genomes.detach().cpu(), values, strict=True):
            self.genomes.append(genome.clone())
            self.values.append([float(v) for v in value])
        if self.max_size is not None and len(self.genomes) > self.max_size:
            ranked_idx = sorted(
                range(len(self.genomes)), key=lambda i: tuple(self.values[i])
            )
            keep_idx = set(ranked_idx[: self.max_size])
            self.genomes = [g for i, g in enumerate(self.genomes) if i in keep_idx]
            self.values = [v for i, v in enumerate(self.values) if i in keep_idx]

    def __len__(self) -> int:
        return len(self.genomes)

    def sample_elite(
        self,
        k: int,
        *,
        pool_size: int | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(self.genomes) == 0:
            raise RuntimeError("Cannot sample from an empty genome archive")
        ranked_idx = sorted(
            range(len(self.genomes)), key=lambda i: tuple(self.values[i])
        )
        pool = ranked_idx[: min(len(ranked_idx), pool_size or len(ranked_idx))]
        k = min(k, len(pool))
        chosen = torch.randperm(len(pool))[:k].tolist()
        return torch.stack([self.genomes[pool[i]] for i in chosen]).to(device, dtype)


class PlackettLuceRankerOpt(BaseOpt):
    """GFog variant that trains D as a listwise ranker over evaluated samples."""

    def __init__(
        self,
        components: components.OptComponents,
        *,
        archive_size: int | None = None,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        generator_elite_margin: bool = False,
    ) -> None:
        self.archive = EvaluatedArchive(max_size=archive_size)
        self.ranker_list_size = ranker_list_size
        self.ranker_steps = ranker_steps
        self.generator_elite_margin = generator_elite_margin
        super().__init__(components)

    def evaluate(self, proposals: torch.Tensor) -> None:
        proposals, keep_idx = self._reject_exact_buffer_design_duplicates(
            proposals.detach()
        )
        if (
            keep_idx is not None
            and self._pending_genomes is not None
            and self._pending_genomes.shape[0] >= int(keep_idx.numel())
        ):
            self._pending_genomes = self._pending_genomes.index_select(
                0,
                keep_idx.to(self._pending_genomes.device),
            )
        if proposals.shape[0] == 0:
            self._last_evaluated_tensors = None
            self._last_evaluated_values = []
            self._pending_genomes = None
            return
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        value_list = [list(v) for v in values]
        detached = proposals.detach()
        self.buffer.B.insert_many(values=value_list, tensors=list(detached))
        self.archive.add_many(detached, value_list)

    def _train_ranker_step(self) -> None:
        if len(self.archive) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self.archive.sample_ranked(
            self.ranker_list_size,
            device=self.gan.device,
            dtype=self.gan.dtype,
        )
        scores = self.gan.D(ranked).reshape(-1)
        loss = plackett_luce_loss(scores)
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals).reshape(-1)
        loss_g = -scores.mean()
        if self.generator_elite_margin and len(self.buffer.B) > 0:
            elite = self.buffer.B.get_top_k(
                min(proposals.shape[0], len(self.buffer.B))
            ).to(self.gan.device, self.gan.dtype)
            elite_scores = self.gan.D(elite).reshape(-1).detach()
            loss_g = F.softplus(-(scores[: elite_scores.numel()] - elite_scores)).mean()
        loss = loss_g
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals


class BufferPlackettLuceRankerOpt(BaseOpt):
    """Listwise ranker optimizer using only the current elite buffer."""

    def __init__(
        self,
        components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        generator_elite_margin: bool = False,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        self.ranker_list_size = ranker_list_size
        self.ranker_steps = ranker_steps
        self.generator_elite_margin = generator_elite_margin
        self.genome_archive = GenomeArchive(
            max_size=components.buffer.B.buffer_size * 4
        )
        self._pending_genomes: torch.Tensor | None = None
        self.ranker_sample_pool_size = ranker_sample_pool_size
        if ranker_sample_mode not in {"random_top_pool", "top_k"}:
            raise ValueError(
                "ranker_sample_mode must be one of random_top_pool, top_k; "
                f"got {ranker_sample_mode}"
            )
        self.ranker_sample_mode = ranker_sample_mode
        self.proposal_pool_size = proposal_pool_size
        self.proposal_top_k = proposal_top_k
        if proposal_diversity_min_hamming < 0:
            raise ValueError(
                "proposal_diversity_min_hamming must be non-negative, "
                f"got {proposal_diversity_min_hamming}"
            )
        if not 0.0 < proposal_diversity_topk_frac < 1.0:
            raise ValueError(
                "proposal_diversity_topk_frac must be in (0, 1), "
                f"got {proposal_diversity_topk_frac}"
            )
        self.proposal_diversity_min_hamming = proposal_diversity_min_hamming
        self.proposal_diversity_topk_frac = proposal_diversity_topk_frac
        if proposal_buffer_novelty_min_hamming < 0:
            raise ValueError(
                "proposal_buffer_novelty_min_hamming must be non-negative, "
                f"got {proposal_buffer_novelty_min_hamming}"
            )
        if proposal_buffer_novelty_reference_size < 1:
            raise ValueError(
                "proposal_buffer_novelty_reference_size must be >= 1, "
                f"got {proposal_buffer_novelty_reference_size}"
            )
        self.proposal_buffer_novelty_min_hamming = proposal_buffer_novelty_min_hamming
        self.proposal_buffer_novelty_reference_size = (
            proposal_buffer_novelty_reference_size
        )
        self.proposal_buffer_reject_exact_design_duplicates = (
            proposal_buffer_reject_exact_design_duplicates
        )
        self.design_proxy = design_proxy
        if not 0.0 <= proposal_evolution_fraction <= 1.0:
            raise ValueError(
                "proposal_evolution_fraction must be in [0, 1], "
                f"got {proposal_evolution_fraction}"
            )
        if proposal_evolution_parent_source not in {"pool", "pool_buffer"}:
            raise ValueError(
                "proposal_evolution_parent_source must be one of pool, pool_buffer; "
                f"got {proposal_evolution_parent_source}"
            )
        if proposal_evolution_crossover not in {"uniform", "row", "rect"}:
            raise ValueError(
                "proposal_evolution_crossover must be one of uniform, row, rect; "
                f"got {proposal_evolution_crossover}"
            )
        if not 0.0 <= proposal_evolution_mutation_rate <= 1.0:
            raise ValueError(
                "proposal_evolution_mutation_rate must be in [0, 1], "
                f"got {proposal_evolution_mutation_rate}"
            )
        if proposal_evolution_mutation_scale < 0:
            raise ValueError(
                "proposal_evolution_mutation_scale must be non-negative, "
                f"got {proposal_evolution_mutation_scale}"
            )
        self.proposal_evolution_fraction = proposal_evolution_fraction
        self.proposal_evolution_parent_source = proposal_evolution_parent_source
        self.proposal_evolution_crossover = proposal_evolution_crossover
        self.proposal_evolution_mutation_rate = proposal_evolution_mutation_rate
        self.proposal_evolution_mutation_scale = proposal_evolution_mutation_scale
        self.proposal_evolution_grid_height = proposal_evolution_grid_height
        self.proposal_evolution_grid_width = proposal_evolution_grid_width
        if proposal_gradient_steps < 0:
            raise ValueError(
                "proposal_gradient_steps must be non-negative, "
                f"got {proposal_gradient_steps}"
            )
        if proposal_gradient_step_size < 0:
            raise ValueError(
                "proposal_gradient_step_size must be non-negative, "
                f"got {proposal_gradient_step_size}"
            )
        if proposal_gradient_noise < 0:
            raise ValueError(
                "proposal_gradient_noise must be non-negative, "
                f"got {proposal_gradient_noise}"
            )
        if proposal_gradient_mode not in {"continuous", "swap"}:
            raise ValueError(
                "proposal_gradient_mode must be one of continuous, swap; "
                f"got {proposal_gradient_mode}"
            )
        self.proposal_gradient_steps = proposal_gradient_steps
        self.proposal_gradient_step_size = proposal_gradient_step_size
        self.proposal_gradient_mode = proposal_gradient_mode
        self.proposal_gradient_normalize = proposal_gradient_normalize
        self.proposal_gradient_noise = proposal_gradient_noise
        self.proposal_gradient_keep_original = proposal_gradient_keep_original
        if not 0.0 <= ga_offspring_fraction <= 1.0:
            raise ValueError(
                f"ga_offspring_fraction must be in [0, 1], got {ga_offspring_fraction}"
            )
        if ga_parent_pool_size < 2:
            raise ValueError(
                f"ga_parent_pool_size must be >= 2, got {ga_parent_pool_size}"
            )
        if not 0.0 <= ga_mutation_rate <= 1.0:
            raise ValueError(
                f"ga_mutation_rate must be in [0, 1], got {ga_mutation_rate}"
            )
        if ga_mutation_scale < 0:
            raise ValueError(
                f"ga_mutation_scale must be non-negative, got {ga_mutation_scale}"
            )
        self.ga_offspring_fraction = ga_offspring_fraction
        self.ga_pool_size = ga_pool_size
        self.ga_parent_pool_size = ga_parent_pool_size
        self.ga_mutation_rate = ga_mutation_rate
        self.ga_mutation_scale = ga_mutation_scale
        if generator_elite_context_size < 0:
            raise ValueError(
                "generator_elite_context_size must be non-negative, "
                f"got {generator_elite_context_size}"
            )
        if (
            generator_elite_context_pool_size is not None
            and generator_elite_context_pool_size < 2
        ):
            raise ValueError(
                "generator_elite_context_pool_size must be >= 2 when set, "
                f"got {generator_elite_context_pool_size}"
            )
        self.generator_elite_context_size = generator_elite_context_size
        self.generator_elite_context_pool_size = generator_elite_context_pool_size
        super().__init__(components)

    def _set_generator_elite_context(self) -> None:
        if not hasattr(self.gan.G, "set_elite_context"):
            return
        if self.generator_elite_context_size <= 0 or len(self.genome_archive) < 2:
            self.gan.G.set_elite_context(None)
            return
        context = self.genome_archive.sample_elite(
            self.generator_elite_context_size,
            pool_size=self.generator_elite_context_pool_size,
            device=self.gan.device,
            dtype=self.gan.dtype,
        )
        self.gan.G.set_elite_context(context)

    def _clear_generator_elite_context(self) -> None:
        if hasattr(self.gan.G, "set_elite_context"):
            self.gan.G.set_elite_context(None)

    def _generate(
        self,
        z: torch.Tensor,
        *,
        use_elite_context: bool = True,
        track_genomes: bool = False,
    ) -> torch.Tensor:
        if use_elite_context:
            self._set_generator_elite_context()
        else:
            self._clear_generator_elite_context()
        try:
            proposals = self.gan.G(z)
            if track_genomes and hasattr(self.gan.G, "last_genomes"):
                last_genomes = self.gan.G.last_genomes
                self._pending_genomes = (
                    None if last_genomes is None else last_genomes.detach()
                )
            elif track_genomes:
                self._pending_genomes = None
            return proposals
        finally:
            self._clear_generator_elite_context()

    def _ranked_buffer_subset_with_positions(
        self, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_len = len(self.buffer.B)
        if current_len == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        if self.ranker_sample_mode == "top_k":
            k = min(k, current_len)
            ranked = self.buffer.B.get_top_k(k)
            positions = torch.arange(
                k,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
            return ranked.to(self.gan.device, self.gan.dtype), positions
        pool_size = current_len
        if self.ranker_sample_pool_size is not None:
            pool_size = min(current_len, max(k, self.ranker_sample_pool_size))
        k = min(k, pool_size)
        if k == pool_size:
            ranked = self.buffer.B.get_top_k(k)
            positions = torch.arange(
                k,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
        else:
            # Buffer indices are already sorted best-to-worst, so sampled sorted
            # positions preserve the true lexicographic order.
            position_list = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.B.get(int(pos)) for pos in position_list])
            positions = torch.as_tensor(
                position_list,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
        return ranked.to(self.gan.device, self.gan.dtype), positions

    def _ranked_buffer_subset(self, k: int) -> torch.Tensor:
        ranked, _ = self._ranked_buffer_subset_with_positions(k)
        return ranked

    def _sample_latents(self, batch_size: int) -> torch.Tensor:
        latent_dim = self.gan.latent_dim
        if not isinstance(latent_dim, int):
            raise ValueError(
                "proposal pool selection currently requires int latent_dim"
            )
        return torch.randn(
            batch_size,
            latent_dim,
            device=self.gan.device,
            dtype=self.gan.dtype,
        )

    def _binary_order_proxy(self, candidates: torch.Tensor) -> torch.Tensor:
        flat = candidates.reshape(candidates.shape[0], -1)
        k = int(round(self.proposal_diversity_topk_frac * flat.shape[1]))
        k = min(max(k, 1), flat.shape[1])
        top_idx = torch.topk(flat, k=k, dim=1).indices
        proxy = torch.zeros_like(flat)
        proxy.scatter_(1, top_idx, 1.0)
        return proxy

    def _design_proxy(self, candidates: torch.Tensor) -> torch.Tensor:
        if self.design_proxy is None:
            return self._binary_order_proxy(candidates).to(torch.bool)
        proxy = self.design_proxy(candidates.detach().cpu())
        if proxy.shape[0] != candidates.shape[0]:
            raise ValueError(
                "design_proxy must return one proxy per candidate, got "
                f"{proxy.shape[0]} for {candidates.shape[0]}"
            )
        return proxy.reshape(proxy.shape[0], -1).to(torch.bool)

    def _buffer_design_proxy(self) -> torch.Tensor | None:
        if (
            (
                self.proposal_buffer_novelty_min_hamming <= 0
                and not self.proposal_buffer_reject_exact_design_duplicates
            )
            or self.design_proxy is None
            or len(self.buffer.B) == 0
        ):
            return None
        reference_size = min(
            self.proposal_buffer_novelty_reference_size,
            len(self.buffer.B),
        )
        reference = self.buffer.B.get_top_k(reference_size).detach().cpu()
        return self._design_proxy(reference)

    def _select_diverse_ranked_candidates(
        self, candidates: torch.Tensor
    ) -> torch.Tensor:
        batch_size = self.components.batch_size
        if candidates.shape[0] <= batch_size:
            return candidates
        if (
            self.proposal_diversity_min_hamming <= 0
            and self.proposal_buffer_novelty_min_hamming <= 0
            and not self.proposal_buffer_reject_exact_design_duplicates
        ):
            return candidates[:batch_size]

        proxy = self._design_proxy(candidates)
        buffer_proxy = self._buffer_design_proxy()
        selected: list[int] = []
        selected_mask = torch.zeros(candidates.shape[0], dtype=torch.bool)
        buffer_rejected_mask = torch.zeros(candidates.shape[0], dtype=torch.bool)
        for idx in range(candidates.shape[0]):
            if buffer_proxy is not None:
                if self.proposal_buffer_reject_exact_design_duplicates and bool(
                    (buffer_proxy == proxy[idx]).all(dim=1).any().item()
                ):
                    buffer_rejected_mask[idx] = True
                    continue
                if self.proposal_buffer_novelty_min_hamming > 0:
                    buffer_distances = (
                        (buffer_proxy != proxy[idx]).to(torch.float32).mean(dim=1)
                    )
                    if (
                        float(buffer_distances.min().item())
                        < self.proposal_buffer_novelty_min_hamming
                    ):
                        buffer_rejected_mask[idx] = True
                        continue
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
            else:
                if self.proposal_diversity_min_hamming <= 0:
                    selected.append(idx)
                    selected_mask[idx] = True
                else:
                    selected_proxy = proxy[selected]
                    distances = (
                        (selected_proxy != proxy[idx]).to(torch.float32).mean(dim=1)
                    )
                    if (
                        float(distances.min().item())
                        >= self.proposal_diversity_min_hamming
                    ):
                        selected.append(idx)
                        selected_mask[idx] = True
            if len(selected) >= batch_size:
                break

        if len(selected) < batch_size:
            for idx in range(candidates.shape[0]):
                if not bool(selected_mask[idx]) and not bool(buffer_rejected_mask[idx]):
                    selected.append(idx)
                if len(selected) >= batch_size:
                    break
        if not selected:
            return candidates[:batch_size]
        return candidates[selected[:batch_size]]

    def _proposal_evolution_parent_pool(self, pool: torch.Tensor) -> torch.Tensor:
        if (
            self.proposal_evolution_parent_source == "pool_buffer"
            and len(self.buffer.B) >= 2
        ):
            parent_pool_size = min(self.ga_parent_pool_size, len(self.buffer.B))
            buffer_parents = self.buffer.B.get_top_k(parent_pool_size).to(
                self.gan.device, self.gan.dtype
            )
            return torch.cat([pool, buffer_parents], dim=0)
        return pool

    def _proposal_evolution_crossover_children(
        self,
        parent_a: torch.Tensor,
        parent_b: torch.Tensor,
    ) -> torch.Tensor:
        if self.proposal_evolution_crossover == "uniform":
            mix_mask = torch.rand_like(parent_a) < 0.5
            return torch.where(mix_mask, parent_a, parent_b)

        height = self.proposal_evolution_grid_height
        width = self.proposal_evolution_grid_width
        if height is None or width is None or parent_a.shape[1] != height * width:
            mix_mask = torch.rand_like(parent_a) < 0.5
            return torch.where(mix_mask, parent_a, parent_b)

        a_grid = parent_a.reshape(parent_a.shape[0], height, width)
        b_grid = parent_b.reshape(parent_b.shape[0], height, width)
        children = a_grid.clone()
        if self.proposal_evolution_crossover == "row":
            row_mask = (
                torch.rand(parent_a.shape[0], height, 1, device=parent_a.device) < 0.5
            )
            children = torch.where(row_mask, a_grid, b_grid)
        else:
            for idx in range(parent_a.shape[0]):
                y0 = int(torch.randint(height, (1,), device=parent_a.device).item())
                y1 = int(
                    torch.randint(
                        y0 + 1, height + 1, (1,), device=parent_a.device
                    ).item()
                )
                x0 = int(torch.randint(width, (1,), device=parent_a.device).item())
                x1 = int(
                    torch.randint(
                        x0 + 1, width + 1, (1,), device=parent_a.device
                    ).item()
                )
                children[idx, y0:y1, x0:x1] = b_grid[idx, y0:y1, x0:x1]
        return children.reshape(parent_a.shape)

    def _augment_proposal_pool_with_evolution(self, pool: torch.Tensor) -> torch.Tensor:
        if self.proposal_evolution_fraction <= 0 or pool.shape[0] < 2:
            return pool
        child_count = int(round(pool.shape[0] * self.proposal_evolution_fraction))
        child_count = min(max(child_count, 0), pool.shape[0])
        if child_count == 0:
            return pool

        parents = self._proposal_evolution_parent_pool(pool)
        parent_count = parents.shape[0]
        parent_a = parents[
            torch.randint(parent_count, (child_count,), device=pool.device)
        ]
        parent_b = parents[
            torch.randint(parent_count, (child_count,), device=pool.device)
        ]
        children = self._proposal_evolution_crossover_children(parent_a, parent_b)
        if (
            self.proposal_evolution_mutation_rate > 0
            and self.proposal_evolution_mutation_scale > 0
        ):
            mutation_mask = (
                torch.rand_like(children) < self.proposal_evolution_mutation_rate
            )
            noise = torch.randn_like(children) * self.proposal_evolution_mutation_scale
            children = torch.where(mutation_mask, children + noise, children)
        if child_count == pool.shape[0]:
            return children
        return torch.cat([pool[: pool.shape[0] - child_count], children], dim=0)

    def _refine_proposals_with_d_gradient(
        self, proposals: torch.Tensor
    ) -> torch.Tensor:
        if self.proposal_gradient_steps <= 0 or self.proposal_gradient_step_size <= 0:
            return proposals.detach()
        if self.proposal_gradient_mode == "swap":
            return self._refine_proposals_with_d_gradient_swaps(proposals)

        original = proposals.detach()
        refined = original.clone()
        for _ in range(self.proposal_gradient_steps):
            refined = refined.detach().requires_grad_(True)
            scores = self.gan.D(refined).reshape(-1)
            grad = torch.autograd.grad(scores.sum(), refined, only_inputs=True)[0]
            if self.proposal_gradient_normalize:
                grad_scale = grad.reshape(grad.shape[0], -1).norm(dim=1).clamp_min(1e-8)
                grad = grad / grad_scale.reshape(-1, *([1] * (grad.ndim - 1)))
            refined = refined + self.proposal_gradient_step_size * grad
            if self.proposal_gradient_noise > 0:
                refined = (
                    refined + torch.randn_like(refined) * self.proposal_gradient_noise
                )

        refined = refined.detach()
        if self.proposal_gradient_keep_original:
            return torch.cat([original, refined], dim=0)
        return refined

    def _refine_proposals_with_d_gradient_swaps(
        self,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        original = proposals.detach()
        refined = original.clone()
        flat_dim = refined.reshape(refined.shape[0], -1).shape[1]
        material_count = int(round(self.proposal_diversity_topk_frac * flat_dim))
        material_count = min(max(material_count, 1), flat_dim - 1)
        swap_count = int(round(self.proposal_gradient_step_size * material_count))
        swap_count = min(max(swap_count, 1), material_count, flat_dim - material_count)

        for _ in range(self.proposal_gradient_steps):
            refined = refined.detach().requires_grad_(True)
            scores = self.gan.D(refined).reshape(-1)
            grad = torch.autograd.grad(scores.sum(), refined, only_inputs=True)[0]
            flat = refined.detach().reshape(refined.shape[0], -1)
            grad_flat = grad.detach().reshape(grad.shape[0], -1)

            top_idx = torch.topk(flat, k=material_count, dim=1).indices
            solid_mask = torch.zeros_like(flat, dtype=torch.bool)
            solid_mask.scatter_(1, top_idx, True)
            void_mask = ~solid_mask

            demote_scores = torch.where(solid_mask, -grad_flat, -torch.inf)
            promote_scores = torch.where(void_mask, grad_flat, -torch.inf)
            demote_idx = torch.topk(demote_scores, k=swap_count, dim=1).indices
            promote_idx = torch.topk(promote_scores, k=swap_count, dim=1).indices

            child = flat.clone()
            demote_values = torch.gather(child, 1, demote_idx)
            promote_values = torch.gather(child, 1, promote_idx)
            margin = torch.clamp(
                (demote_values - promote_values).abs() + 1e-3,
                min=1e-3,
            )
            child.scatter_(1, promote_idx, demote_values + margin)
            child.scatter_(1, demote_idx, promote_values - margin)
            if self.proposal_gradient_noise > 0:
                child = child + torch.randn_like(child) * self.proposal_gradient_noise
            refined = child.reshape_as(refined)

        refined = refined.detach()
        if self.proposal_gradient_keep_original:
            return torch.cat([original, refined], dim=0)
        return refined

    def _select_ranked_exploration_proposals(
        self, proposals: torch.Tensor
    ) -> torch.Tensor:
        if self.proposal_pool_size is None:
            refined = self._refine_proposals_with_d_gradient(proposals)
            if refined.shape[0] > self.components.batch_size:
                with torch.no_grad():
                    scores = self.gan.D(refined).reshape(-1)
                    order = torch.topk(
                        scores,
                        k=self.components.batch_size,
                        dim=0,
                    ).indices
                    refined = refined[order]
            return self._select_ga_mixed_proposals(refined)

        pool_size = max(self.proposal_pool_size, self.components.batch_size)
        with torch.no_grad():
            z = self._sample_latents(pool_size)
            pool = self._generate(z)
            pool = self._augment_proposal_pool_with_evolution(pool)
        pool = self._refine_proposals_with_d_gradient(pool)
        with torch.no_grad():
            scores = self.gan.D(pool).reshape(-1)
            top_k = self.proposal_top_k or pool_size
            top_k = min(max(top_k, self.components.batch_size), pool.shape[0])
            order = torch.topk(scores, k=top_k, dim=0).indices
            ranked_pool = pool[order]
            selected = self._select_diverse_ranked_candidates(ranked_pool)
        return self._select_ga_mixed_proposals(selected)

    def _make_ga_children(self, count: int) -> torch.Tensor:
        parent_pool_size = min(self.ga_parent_pool_size, len(self.buffer.B))
        parents = self.buffer.B.get_top_k(parent_pool_size).to(
            self.gan.device, self.gan.dtype
        )
        parent_a = parents[
            torch.randint(parent_pool_size, (count,), device=self.gan.device)
        ]
        parent_b = parents[
            torch.randint(parent_pool_size, (count,), device=self.gan.device)
        ]
        mix_mask = torch.rand_like(parent_a) < 0.5
        children = torch.where(mix_mask, parent_a, parent_b)
        if self.ga_mutation_rate > 0 and self.ga_mutation_scale > 0:
            mutation_mask = torch.rand_like(children) < self.ga_mutation_rate
            noise = torch.randn_like(children) * self.ga_mutation_scale
            children = torch.where(mutation_mask, children + noise, children)
        return children

    def _select_ga_children(self, count: int) -> torch.Tensor:
        pool_size = self.ga_pool_size or max(count * 4, self.components.batch_size)
        pool_size = max(pool_size, count)
        children = self._make_ga_children(pool_size)
        scores = self.gan.D(children).reshape(-1)
        order = torch.topk(scores, k=count, dim=0).indices
        return children[order]

    def _select_ga_mixed_proposals(self, proposals: torch.Tensor) -> torch.Tensor:
        if self.ga_offspring_fraction <= 0 or len(self.buffer.B) < 2:
            return proposals
        batch_size = proposals.shape[0]
        ga_count = int(round(batch_size * self.ga_offspring_fraction))
        ga_count = min(max(ga_count, 0), batch_size)
        if ga_count == 0:
            return proposals
        with torch.no_grad():
            ga_children = self._select_ga_children(ga_count)
        if ga_count == batch_size:
            return ga_children
        return torch.cat(
            [proposals[: batch_size - ga_count].detach(), ga_children], dim=0
        )

    def _reject_exact_buffer_design_duplicates(
        self,
        proposals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not self.proposal_buffer_reject_exact_design_duplicates
            or self.design_proxy is None
            or len(self.buffer.B) == 0
            or proposals.shape[0] == 0
        ):
            return proposals, None

        buffer_proxy = self._buffer_design_proxy()
        if buffer_proxy is None:
            return proposals, None

        proposal_proxy = self._design_proxy(proposals)
        keep: list[int] = []
        for idx in range(proposal_proxy.shape[0]):
            candidate = proposal_proxy[idx]
            in_buffer = bool((buffer_proxy == candidate).all(dim=1).any().item())
            if in_buffer:
                continue
            if keep:
                kept_proxy = proposal_proxy[keep]
                in_batch = bool((kept_proxy == candidate).all(dim=1).any().item())
                if in_batch:
                    continue
            keep.append(idx)

        if len(keep) == proposals.shape[0]:
            return proposals, None
        if not keep:
            return proposals[:0], torch.empty(0, dtype=torch.long)
        keep_idx = torch.as_tensor(keep, dtype=torch.long, device=proposals.device)
        return proposals.index_select(0, keep_idx), keep_idx.detach().cpu()

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self._ranked_buffer_subset(self.ranker_list_size)
        scores = self.gan.D(ranked).reshape(-1)
        loss = plackett_luce_loss(scores)
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        scores = self.gan.D(proposals).reshape(-1)
        if self.generator_elite_margin:
            elite = self._ranked_buffer_subset(
                min(proposals.shape[0], len(self.buffer.B))
            )
            elite_scores = self.gan.D(elite).reshape(-1).detach()
            loss_g = F.softplus(-(scores[: elite_scores.numel()] - elite_scores)).mean()
        else:
            loss_g = -scores.mean()
        loss = loss_g
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals

    def evaluate(self, proposals: torch.Tensor) -> None:
        proposals, keep_idx = self._reject_exact_buffer_design_duplicates(
            proposals.detach()
        )
        if (
            keep_idx is not None
            and self._pending_genomes is not None
            and self._pending_genomes.shape[0] >= int(keep_idx.numel())
        ):
            self._pending_genomes = self._pending_genomes.index_select(
                0,
                keep_idx.to(self._pending_genomes.device),
            )
        if proposals.shape[0] == 0:
            self._pending_genomes = None
            return
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        value_list = [list(v) for v in values]
        self.buffer.B.insert_many(values=value_list, tensors=list(proposals.detach()))
        if (
            self._pending_genomes is not None
            and self._pending_genomes.shape[0] == proposals.shape[0]
        ):
            self.genome_archive.add_many(self._pending_genomes, value_list)
        self._pending_genomes = None


class ContextualPlackettLuceRankerOpt(BufferPlackettLuceRankerOpt):
    """PL ranker whose generator is trained against an evaluated context list."""

    def __init__(
        self,
        components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        d_score_center_weight: float = 0.0,
        d_score_scale_weight: float = 0.0,
        d_score_target_std: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if d_score_center_weight < 0:
            raise ValueError(
                "d_score_center_weight must be non-negative, "
                f"got {d_score_center_weight}"
            )
        if d_score_scale_weight < 0:
            raise ValueError(
                f"d_score_scale_weight must be non-negative, got {d_score_scale_weight}"
            )
        if d_score_target_std <= 0:
            raise ValueError(
                f"d_score_target_std must be positive, got {d_score_target_std}"
            )
        self.d_score_center_weight = d_score_center_weight
        self.d_score_scale_weight = d_score_scale_weight
        self.d_score_target_std = d_score_target_std
        self._last_evaluated_tensors: torch.Tensor | None = None
        self._last_evaluated_values: list[list[float]] = []
        super().__init__(
            components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            **kwargs,
        )

    def _sample_ranked_mixed_evaluated_list(self, k: int) -> torch.Tensor:
        if self._last_evaluated_tensors is None or not self._last_evaluated_values:
            return self._ranked_buffer_subset(k)

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
        items.extend(
            (
                value,
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
        return torch.stack([tensor for _value, tensor in items]).to(
            self.gan.device,
            self.gan.dtype,
        )

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        ranked = self._sample_ranked_mixed_evaluated_list(self.ranker_list_size)
        scores = self.gan.D(ranked).reshape(-1)
        loss = plackett_luce_loss(scores)
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
        proposals = self._generate(z, track_genomes=True)
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
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)

    def evaluate(self, proposals: torch.Tensor) -> None:
        proposals, keep_idx = self._reject_exact_buffer_design_duplicates(
            proposals.detach()
        )
        if (
            keep_idx is not None
            and self._pending_genomes is not None
            and self._pending_genomes.shape[0] >= int(keep_idx.numel())
        ):
            self._pending_genomes = self._pending_genomes.index_select(
                0,
                keep_idx.to(self._pending_genomes.device),
            )
        if proposals.shape[0] == 0:
            self._last_evaluated_tensors = None
            self._last_evaluated_values = []
            self._pending_genomes = None
            return
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        value_list = [list(v) for v in values]
        detached = proposals.detach()
        self.buffer.B.insert_many(values=value_list, tensors=list(detached))
        self._last_evaluated_tensors = detached.cpu()
        self._last_evaluated_values = value_list
        if (
            self._pending_genomes is not None
            and self._pending_genomes.shape[0] == proposals.shape[0]
        ):
            self.genome_archive.add_many(self._pending_genomes, value_list)
        self._pending_genomes = None


class CalibratedUtilityRankerOpt(ContextualPlackettLuceRankerOpt):
    """Reward model that predicts calibrated utility from true evaluated values."""

    def __init__(
        self,
        components: components.OptComponents,
        *,
        utility_target_scale: float = 100.0,
        utility_loss: str = "smooth_l1",
        **kwargs: Any,
    ) -> None:
        if utility_target_scale <= 0:
            raise ValueError(
                f"utility_target_scale must be positive, got {utility_target_scale}"
            )
        if utility_loss not in {"smooth_l1", "mse"}:
            raise ValueError(
                f"utility_loss must be one of smooth_l1, mse; got {utility_loss}"
            )
        self.utility_target_scale = utility_target_scale
        self.utility_loss = utility_loss
        super().__init__(components, **kwargs)

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
                    value,
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

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()
        tensors, values = self._sample_mixed_evaluated_items(self.ranker_list_size)
        scores = self.gan.D(tensors).reshape(-1)
        targets = utility_targets_from_values(
            values,
            scale=self.utility_target_scale,
            device=scores.device,
            dtype=scores.dtype,
        )
        if self.utility_loss == "mse":
            loss = F.mse_loss(scores, targets)
        else:
            loss = F.smooth_l1_loss(scores, targets)
        if self.d_score_center_weight > 0:
            loss = (
                loss + self.d_score_center_weight * (scores - targets).mean().square()
            )
        if self.d_score_scale_weight > 0 and scores.numel() > 1:
            residual_std = (scores - targets).std(unbiased=False)
            loss = loss + self.d_score_scale_weight * residual_std.square()
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        scores = self.gan.D(proposals).reshape(-1)
        loss = -scores.mean()
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


class HybridContextualUtilityRankerOpt(CalibratedUtilityRankerOpt):
    """Contextual PL ranker with light local utility calibration."""

    def __init__(
        self,
        components: components.OptComponents,
        *,
        utility_weight: float = 0.1,
        generator_utility_weight: float = 0.0,
        utility_clip: float = 3.0,
        **kwargs: Any,
    ) -> None:
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
        self.utility_weight = utility_weight
        self.generator_utility_weight = generator_utility_weight
        self.utility_clip = utility_clip
        super().__init__(components, **kwargs)

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
        proposals = self._generate(z, track_genomes=True)
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
        return self._select_ranked_exploration_proposals(proposals)


class RankedLSGANOpt(BufferPlackettLuceRankerOpt):
    """LSGAN with an additional Plackett-Luce ranking loss on buffer elites."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 0.1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
        if ranker_weight < 0:
            raise ValueError(f"ranker_weight must be non-negative, got {ranker_weight}")
        self.ranker_weight = ranker_weight

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()

        ranked = self._ranked_buffer_subset(self.ranker_list_size)
        real_scores = self.gan.D(ranked).reshape(-1)
        rank_loss = plackett_luce_loss(real_scores)

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            fake = self._generate(z)
        fake_scores = self.gan.D(fake.detach())
        fake_loss = 0.5 * (fake_scores**2).mean()

        loss = fake_loss + self.ranker_weight * rank_loss
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        scores = self.gan.D(proposals)
        loss = 0.5 * ((scores - 1.0) ** 2).mean()
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


class RankedDefaultOpt(BufferPlackettLuceRankerOpt):
    """Vanilla GAN with an auxiliary Plackett-Luce ranking loss on buffer elites."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 0.1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
        if ranker_weight < 0:
            raise ValueError(f"ranker_weight must be non-negative, got {ranker_weight}")
        self.ranker_weight = ranker_weight

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()

        ranked = self._ranked_buffer_subset(self.ranker_list_size)
        real_scores = self.gan.D(ranked).reshape(-1)
        rank_loss = plackett_luce_loss(real_scores)

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            fake = self._generate(z)
        fake_scores = self.gan.D(fake.detach())
        fake_loss = self.gan.loss(fake_scores, torch.zeros_like(fake_scores))

        loss = fake_loss + self.ranker_weight * rank_loss
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        scores = self.gan.D(proposals)
        loss = self.gan.loss(scores, torch.ones_like(scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


class RankedWGANOpt(BufferPlackettLuceRankerOpt):
    """WGAN with an auxiliary Plackett-Luce ranking loss on buffer elites."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 0.1,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
        if ranker_weight < 0:
            raise ValueError(f"ranker_weight must be non-negative, got {ranker_weight}")
        self.ranker_weight = ranker_weight

    def _apply_weight_clipping(self) -> None:
        if self.components.weight_clip is None:
            return
        for parameter in self.gan.D.parameters():
            parameter.data.clamp_(
                -self.components.weight_clip,
                self.components.weight_clip,
            )

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()

        ranked = self._ranked_buffer_subset(self.ranker_list_size)
        real_scores = self.gan.D(ranked).reshape(-1)
        rank_loss = plackett_luce_loss(real_scores)

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            fake = self._generate(z)
        fake_scores = self.gan.D(fake.detach()).reshape(-1)

        loss = fake_scores.mean() + self.ranker_weight * rank_loss
        loss.backward()
        self.gan.optimizerD.step()
        self._apply_weight_clipping()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        loss = -self.gan.D(proposals).reshape(-1).mean()
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


class QuantileRankedDefaultOpt(BufferPlackettLuceRankerOpt):
    """Vanilla GAN with rank-quantile targets for current buffer elites."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 1.0,
        ranker_target_curve: str = "linear",
        ranker_tau: float = 16.0,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        ranker_target_scope: str = "local",
        ranker_niche_local_targets: bool = False,
        ranker_list_repeats: int = 1,
        ranker_fake_weight: float = 1.0,
        ranker_fake_repeats: int = 1,
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        if ranker_niche_local_targets:
            niche_buffer = opt_components.buffer.B
            required_methods = (
                "niche_count",
                "get_niche",
                "get_niche_length",
                "get_niche_top_k",
            )
            missing = [
                name for name in required_methods if not hasattr(niche_buffer, name)
            ]
            if missing:
                raise ValueError(
                    "--ranker_niche_local_targets requires a niche buffer; "
                    f"missing {missing}"
                )
            if ranker_list_size < int(niche_buffer.niche_count):
                raise ValueError(
                    "ranker_list_size must be >= niche_count when "
                    "--ranker_niche_local_targets is set"
                )
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
        if ranker_weight < 0:
            raise ValueError(f"ranker_weight must be non-negative, got {ranker_weight}")
        self.ranker_weight = ranker_weight
        self.ranker_target_curve = ranker_target_curve
        self.ranker_tau = ranker_tau
        if ranker_list_repeats < 1:
            raise ValueError(
                f"ranker_list_repeats must be >= 1, got {ranker_list_repeats}"
            )
        if ranker_fake_weight < 0:
            raise ValueError(
                f"ranker_fake_weight must be non-negative, got {ranker_fake_weight}"
            )
        if ranker_fake_repeats < 1:
            raise ValueError(
                f"ranker_fake_repeats must be >= 1, got {ranker_fake_repeats}"
            )
        self.ranker_list_repeats = ranker_list_repeats
        self.ranker_fake_weight = ranker_fake_weight
        self.ranker_fake_repeats = ranker_fake_repeats
        if ranker_target_scope not in {"local", "global"}:
            raise ValueError(
                "ranker_target_scope must be one of local, global; "
                f"got {ranker_target_scope}"
            )
        self.ranker_target_scope = ranker_target_scope
        self.ranker_niche_local_targets = ranker_niche_local_targets

    def _ranked_niche_buffer_subset_with_positions(
        self,
        niche_idx: int,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_len = self.buffer.B.get_niche_length(niche_idx)
        if current_len == 0:
            raise RuntimeError(f"Cannot sample from empty niche {niche_idx}")
        if self.ranker_sample_mode == "top_k":
            k = min(k, current_len)
            ranked = self.buffer.B.get_niche_top_k(niche_idx, k)
            positions = torch.arange(
                k,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
            return ranked.to(self.gan.device, self.gan.dtype), positions

        pool_size = current_len
        if self.ranker_sample_pool_size is not None:
            per_niche_pool_size = math.ceil(
                self.ranker_sample_pool_size / self.buffer.B.niche_count
            )
            pool_size = min(current_len, max(k, per_niche_pool_size))
        k = min(k, pool_size)
        if k == pool_size:
            ranked = self.buffer.B.get_niche_top_k(niche_idx, k)
            positions = torch.arange(
                k,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
        else:
            position_list = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack(
                [self.buffer.B.get_niche(niche_idx, int(pos)) for pos in position_list]
            )
            positions = torch.as_tensor(
                position_list,
                device=self.gan.device,
                dtype=self.gan.dtype,
            )
        return ranked.to(self.gan.device, self.gan.dtype), positions

    def _niche_local_real_rank_loss(self) -> torch.Tensor:
        niche_count = int(self.buffer.B.niche_count)
        list_sizes = niche_capacities(self.ranker_list_size, niche_count)
        repeat_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        active_repeats = 0
        for _ in range(self.ranker_list_repeats):
            niche_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
            active_niches = 0
            for niche_idx, list_size in enumerate(list_sizes):
                if self.buffer.B.get_niche_length(niche_idx) < 2:
                    continue
                ranked, positions = self._ranked_niche_buffer_subset_with_positions(
                    niche_idx,
                    list_size,
                )
                real_scores = self.gan.D(ranked)
                if self.ranker_target_scope == "global":
                    real_targets = rank_targets(
                        self.buffer.B.get_niche_length(niche_idx),
                        device=real_scores.device,
                        dtype=real_scores.dtype,
                        curve=self.ranker_target_curve,
                        tau=self.ranker_tau,
                    )[positions.long()].reshape_as(real_scores)
                else:
                    real_targets = rank_targets(
                        real_scores.numel(),
                        device=real_scores.device,
                        dtype=real_scores.dtype,
                        curve=self.ranker_target_curve,
                        tau=self.ranker_tau,
                    ).reshape_as(real_scores)
                niche_loss = niche_loss + self.gan.loss(real_scores, real_targets)
                active_niches += 1
            if active_niches > 0:
                repeat_loss = repeat_loss + niche_loss / active_niches
                active_repeats += 1
        if active_repeats == 0:
            return repeat_loss
        return repeat_loss / active_repeats

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()

        if self.ranker_niche_local_targets:
            real_loss = self._niche_local_real_rank_loss()
        else:
            real_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
            for _ in range(self.ranker_list_repeats):
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
                    )[positions.long()].reshape_as(real_scores)
                else:
                    real_targets = rank_targets(
                        real_scores.numel(),
                        device=real_scores.device,
                        dtype=real_scores.dtype,
                        curve=self.ranker_target_curve,
                        tau=self.ranker_tau,
                    ).reshape_as(real_scores)
                real_loss = real_loss + self.gan.loss(real_scores, real_targets)
            real_loss = real_loss / self.ranker_list_repeats

        fake_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        if self.ranker_fake_weight > 0:
            for _ in range(self.ranker_fake_repeats):
                with torch.no_grad():
                    z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
                    fake = self._generate(z)
                fake_scores = self.gan.D(fake.detach())
                fake_loss = fake_loss + self.gan.loss(
                    fake_scores,
                    torch.zeros_like(fake_scores),
                )
            fake_loss = fake_loss / self.ranker_fake_repeats

        loss = self.ranker_fake_weight * fake_loss + self.ranker_weight * real_loss
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        scores = self.gan.D(proposals)
        loss = self.gan.loss(scores, torch.ones_like(scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


class QuantileRankedValueDefaultOpt(QuantileRankedDefaultOpt):
    """Quantile ranker with an auxiliary log-compliance value head."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 1.0,
        ranker_target_curve: str = "linear",
        ranker_tau: float = 16.0,
        ranker_sample_pool_size: int | None = None,
        ranker_sample_mode: str = "random_top_pool",
        ranker_target_scope: str = "local",
        ranker_list_repeats: int = 1,
        ranker_fake_weight: float = 1.0,
        ranker_fake_repeats: int = 1,
        utility_target_scale: float = 100.0,
        utility_loss: str = "smooth_l1",
        utility_weight: float = 0.1,
        generator_utility_weight: float = 0.0,
        utility_clip: float = 3.0,
        proposal_pool_size: int | None = None,
        proposal_top_k: int | None = None,
        proposal_diversity_min_hamming: float = 0.0,
        proposal_diversity_topk_frac: float = 0.48,
        proposal_buffer_novelty_min_hamming: float = 0.0,
        proposal_buffer_novelty_reference_size: int = 128,
        proposal_buffer_reject_exact_design_duplicates: bool = False,
        design_proxy: DesignProxyFn | None = None,
        proposal_evolution_fraction: float = 0.0,
        proposal_evolution_parent_source: str = "pool",
        proposal_evolution_crossover: str = "uniform",
        proposal_evolution_mutation_rate: float = 0.01,
        proposal_evolution_mutation_scale: float = 0.10,
        proposal_evolution_grid_height: int | None = None,
        proposal_evolution_grid_width: int | None = None,
        proposal_gradient_steps: int = 0,
        proposal_gradient_step_size: float = 0.05,
        proposal_gradient_mode: str = "continuous",
        proposal_gradient_normalize: bool = True,
        proposal_gradient_noise: float = 0.0,
        proposal_gradient_keep_original: bool = False,
        ga_offspring_fraction: float = 0.0,
        ga_pool_size: int | None = None,
        ga_parent_pool_size: int = 128,
        ga_mutation_rate: float = 0.02,
        ga_mutation_scale: float = 0.25,
        generator_elite_context_size: int = 0,
        generator_elite_context_pool_size: int | None = None,
    ) -> None:
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_target_curve=ranker_target_curve,
            ranker_tau=ranker_tau,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            ranker_target_scope=ranker_target_scope,
            ranker_list_repeats=ranker_list_repeats,
            ranker_fake_weight=ranker_fake_weight,
            ranker_fake_repeats=ranker_fake_repeats,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
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
        if generator_utility_weight != 0:
            raise ValueError(
                "quantile_ranked_value_default uses utility only as a D-side "
                f"auxiliary loss; generator_utility_weight must be 0, got {generator_utility_weight}"
            )
        if utility_clip < 0:
            raise ValueError(f"utility_clip must be non-negative, got {utility_clip}")
        self.utility_target_scale = utility_target_scale
        self.utility_loss = utility_loss
        self.utility_weight = utility_weight
        self.generator_utility_weight = 0.0
        self.utility_clip = utility_clip

    def _compliance_targets_for_positions(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        compliance = torch.as_tensor(
            [
                self.buffer.B.get_value(int(position), level=-1)
                for position in positions.detach().cpu().tolist()
            ],
            device=self.gan.device,
            dtype=self.gan.dtype,
        )
        return log_compliance_utility_targets(
            compliance,
            reference=self.utility_target_scale,
            clip=self.utility_clip,
        )

    def _value_loss(
        self,
        value_scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        value_scores = value_scores.reshape_as(targets)
        if self.utility_loss == "mse":
            return F.mse_loss(value_scores, targets)
        return F.smooth_l1_loss(value_scores, targets)

    def _train_ranker_step(self) -> None:
        if len(self.buffer.B) < 2:
            return
        self.gan.optimizerD.zero_grad()

        rank_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        value_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        for _ in range(self.ranker_list_repeats):
            ranked, positions = self._ranked_buffer_subset_with_positions(
                self.ranker_list_size
            )
            rank_scores, value_scores = split_rank_value_scores(self.gan.D(ranked))
            if self.ranker_target_scope == "global":
                real_targets = rank_targets(
                    len(self.buffer.B),
                    device=rank_scores.device,
                    dtype=rank_scores.dtype,
                    curve=self.ranker_target_curve,
                    tau=self.ranker_tau,
                )[positions.long()].reshape_as(rank_scores)
            else:
                real_targets = rank_targets(
                    rank_scores.numel(),
                    device=rank_scores.device,
                    dtype=rank_scores.dtype,
                    curve=self.ranker_target_curve,
                    tau=self.ranker_tau,
                ).reshape_as(rank_scores)
            rank_loss = rank_loss + self.gan.loss(rank_scores, real_targets)
            value_targets = self._compliance_targets_for_positions(positions)
            value_loss = value_loss + self._value_loss(
                value_scores.reshape(-1),
                value_targets,
            )
        rank_loss = rank_loss / self.ranker_list_repeats
        value_loss = value_loss / self.ranker_list_repeats

        fake_loss = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        if self.ranker_fake_weight > 0:
            for _ in range(self.ranker_fake_repeats):
                with torch.no_grad():
                    z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
                    fake = self._generate(z)
                fake_rank_scores, _fake_value_scores = split_rank_value_scores(
                    self.gan.D(fake.detach())
                )
                fake_loss = fake_loss + self.gan.loss(
                    fake_rank_scores,
                    torch.zeros_like(fake_rank_scores),
                )
            fake_loss = fake_loss / self.ranker_fake_repeats

        loss = (
            self.ranker_fake_weight * fake_loss
            + self.ranker_weight * rank_loss
            + self.utility_weight * value_loss
        )
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        for _ in range(self.ranker_steps):
            self._train_ranker_step()

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self._generate(z, track_genomes=True)
        rank_scores, _value_scores = split_rank_value_scores(self.gan.D(proposals))
        loss = self.gan.loss(rank_scores, torch.ones_like(rank_scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return self._select_ranked_exploration_proposals(proposals)


def make_optimizer(
    optimizer_type: str,
    opt_components: components.OptComponents,
    *,
    archive_size: int | None = None,
    ranker_list_size: int = 32,
    ranker_steps: int = 1,
    ranker_generator_elite_margin: bool = False,
    ranker_weight: float = 0.1,
    ranker_target_curve: str = "linear",
    ranker_tau: float = 16.0,
    ranker_sample_pool_size: int | None = None,
    ranker_sample_mode: str = "random_top_pool",
    ranker_target_scope: str = "local",
    ranker_niche_local_targets: bool = False,
    ranker_list_repeats: int = 1,
    ranker_fake_weight: float = 1.0,
    ranker_fake_repeats: int = 1,
    proposal_pool_size: int | None = None,
    proposal_top_k: int | None = None,
    proposal_diversity_min_hamming: float = 0.0,
    proposal_diversity_topk_frac: float = 0.48,
    proposal_buffer_novelty_min_hamming: float = 0.0,
    proposal_buffer_novelty_reference_size: int = 128,
    proposal_buffer_reject_exact_design_duplicates: bool = False,
    design_proxy: DesignProxyFn | None = None,
    proposal_evolution_fraction: float = 0.0,
    proposal_evolution_parent_source: str = "pool",
    proposal_evolution_crossover: str = "uniform",
    proposal_evolution_mutation_rate: float = 0.01,
    proposal_evolution_mutation_scale: float = 0.10,
    proposal_evolution_grid_height: int | None = None,
    proposal_evolution_grid_width: int | None = None,
    proposal_gradient_steps: int = 0,
    proposal_gradient_step_size: float = 0.05,
    proposal_gradient_mode: str = "continuous",
    proposal_gradient_normalize: bool = True,
    proposal_gradient_noise: float = 0.0,
    proposal_gradient_keep_original: bool = False,
    ga_offspring_fraction: float = 0.0,
    ga_pool_size: int | None = None,
    ga_parent_pool_size: int = 128,
    ga_mutation_rate: float = 0.02,
    ga_mutation_scale: float = 0.25,
    generator_elite_context_size: int = 0,
    generator_elite_context_pool_size: int | None = None,
    d_score_center_weight: float = 0.0,
    d_score_scale_weight: float = 0.0,
    d_score_target_std: float = 1.0,
    utility_target_scale: float = 100.0,
    utility_loss: str = "smooth_l1",
    utility_weight: float = 0.1,
    generator_utility_weight: float = 0.0,
    utility_clip: float = 3.0,
) -> BaseOpt:
    if optimizer_type == "default":
        return DefaultOpt(opt_components)
    if optimizer_type == "hinge":
        return HingeGANOpt(opt_components)
    if optimizer_type == "lsgan":
        return LSGANOpt(opt_components)
    if optimizer_type == "wgan":
        return WGANOpt(opt_components)
    if optimizer_type == "wgangp":
        return WGANGPOpt(opt_components)
    if optimizer_type == "plackett_luce":
        return PlackettLuceRankerOpt(
            opt_components,
            archive_size=archive_size,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=ranker_generator_elite_margin,
        )
    if optimizer_type == "buffer_plackett_luce":
        return BufferPlackettLuceRankerOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=ranker_generator_elite_margin,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    if optimizer_type == "contextual_plackett_luce":
        return ContextualPlackettLuceRankerOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
            d_score_center_weight=d_score_center_weight,
            d_score_scale_weight=d_score_scale_weight,
            d_score_target_std=d_score_target_std,
        )
    if optimizer_type == "calibrated_utility":
        return CalibratedUtilityRankerOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
            d_score_center_weight=d_score_center_weight,
            d_score_scale_weight=d_score_scale_weight,
            d_score_target_std=d_score_target_std,
            utility_target_scale=utility_target_scale,
            utility_loss=utility_loss,
        )
    if optimizer_type == "hybrid_contextual_utility":
        return HybridContextualUtilityRankerOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
            d_score_center_weight=d_score_center_weight,
            d_score_scale_weight=d_score_scale_weight,
            d_score_target_std=d_score_target_std,
            utility_target_scale=utility_target_scale,
            utility_loss=utility_loss,
            utility_weight=utility_weight,
            generator_utility_weight=generator_utility_weight,
            utility_clip=utility_clip,
        )
    if optimizer_type == "ranked_lsgan":
        return RankedLSGANOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    if optimizer_type == "ranked_default":
        return RankedDefaultOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    if optimizer_type == "ranked_wgan":
        return RankedWGANOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    if optimizer_type == "quantile_ranked_default":
        return QuantileRankedDefaultOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_target_curve=ranker_target_curve,
            ranker_tau=ranker_tau,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            ranker_target_scope=ranker_target_scope,
            ranker_niche_local_targets=ranker_niche_local_targets,
            ranker_list_repeats=ranker_list_repeats,
            ranker_fake_weight=ranker_fake_weight,
            ranker_fake_repeats=ranker_fake_repeats,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    if optimizer_type == "quantile_ranked_value_default":
        return QuantileRankedValueDefaultOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
            ranker_target_curve=ranker_target_curve,
            ranker_tau=ranker_tau,
            ranker_sample_pool_size=ranker_sample_pool_size,
            ranker_sample_mode=ranker_sample_mode,
            ranker_target_scope=ranker_target_scope,
            ranker_list_repeats=ranker_list_repeats,
            ranker_fake_weight=ranker_fake_weight,
            ranker_fake_repeats=ranker_fake_repeats,
            utility_target_scale=utility_target_scale,
            utility_loss=utility_loss,
            utility_weight=utility_weight,
            generator_utility_weight=generator_utility_weight,
            utility_clip=utility_clip,
            proposal_pool_size=proposal_pool_size,
            proposal_top_k=proposal_top_k,
            proposal_diversity_min_hamming=proposal_diversity_min_hamming,
            proposal_diversity_topk_frac=proposal_diversity_topk_frac,
            proposal_buffer_novelty_min_hamming=proposal_buffer_novelty_min_hamming,
            proposal_buffer_novelty_reference_size=proposal_buffer_novelty_reference_size,
            proposal_buffer_reject_exact_design_duplicates=proposal_buffer_reject_exact_design_duplicates,
            design_proxy=design_proxy,
            proposal_evolution_fraction=proposal_evolution_fraction,
            proposal_evolution_parent_source=proposal_evolution_parent_source,
            proposal_evolution_crossover=proposal_evolution_crossover,
            proposal_evolution_mutation_rate=proposal_evolution_mutation_rate,
            proposal_evolution_mutation_scale=proposal_evolution_mutation_scale,
            proposal_evolution_grid_height=proposal_evolution_grid_height,
            proposal_evolution_grid_width=proposal_evolution_grid_width,
            proposal_gradient_steps=proposal_gradient_steps,
            proposal_gradient_step_size=proposal_gradient_step_size,
            proposal_gradient_mode=proposal_gradient_mode,
            proposal_gradient_normalize=proposal_gradient_normalize,
            proposal_gradient_noise=proposal_gradient_noise,
            proposal_gradient_keep_original=proposal_gradient_keep_original,
            ga_offspring_fraction=ga_offspring_fraction,
            ga_pool_size=ga_pool_size,
            ga_parent_pool_size=ga_parent_pool_size,
            ga_mutation_rate=ga_mutation_rate,
            ga_mutation_scale=ga_mutation_scale,
            generator_elite_context_size=generator_elite_context_size,
            generator_elite_context_pool_size=generator_elite_context_pool_size,
        )
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def make_decoded_density_optimizer(
    optimizer_type: str,
    opt_components: components.OptComponents,
    *,
    evaluator: FEMCantileverEvaluator,
    objective: BufferChamferDiversityObjective | None = None,
) -> BaseOpt:
    kwargs = {
        "decoder": evaluator.decode_designs_torch,
        "density_objective": (
            objective.evaluate_densities_numpy
            if objective is not None
            else evaluator.evaluate_densities_numpy
        ),
    }
    if optimizer_type == "default":
        return DecodedDensityDefaultOpt(opt_components, **kwargs)
    if optimizer_type == "hinge":
        return DecodedDensityHingeGANOpt(opt_components, **kwargs)
    if optimizer_type == "lsgan":
        return DecodedDensityLSGANOpt(opt_components, **kwargs)
    if optimizer_type == "wgan":
        return DecodedDensityWGANOpt(opt_components, **kwargs)
    if optimizer_type == "wgangp":
        return DecodedDensityWGANGPOpt(opt_components, **kwargs)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def compliance_summary(
    values: np.ndarray, feasibility_eps: float = 1e-6
) -> dict[str, Any]:
    volume_violation = values[:, -3]
    roughness_violation = values[:, -2]
    compliance = values[:, -1]
    feasible = (volume_violation <= feasibility_eps) & (
        roughness_violation <= feasibility_eps
    )

    best_feasible_compliance = float("nan")
    best_feasible_index = -1
    if np.any(feasible):
        feasible_indices = np.flatnonzero(feasible)
        local_best = int(np.argmin(compliance[feasible]))
        best_feasible_index = int(feasible_indices[local_best])
        best_feasible_compliance = float(compliance[best_feasible_index])

    best_any_index = int(np.argmin(compliance))
    return {
        "archive_best_value": values[0].tolist(),
        "archive_best_compliance": float(compliance[0]),
        "best_feasible_compliance": best_feasible_compliance,
        "best_feasible_index": best_feasible_index,
        "best_any_compliance": float(compliance[best_any_index]),
        "best_any_index": best_any_index,
        "feasible_count": int(np.count_nonzero(feasible)),
        "feasible_rate": float(np.mean(feasible)),
    }


def record_buffer_history(
    buffer: Buffer,
    *,
    iteration: int,
    eval_count: int,
) -> dict[str, np.ndarray | float]:
    """Capture cheap rank-buffer telemetry without extra FEM evaluations."""
    values = np.asarray(buffer.get_sorted_values(), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    last_level = values[:, -1]
    feasible_rate = float("nan")
    best_feasible_last = float("nan")
    mean_volume_violation = float("nan")
    mean_roughness_violation = float("nan")
    mean_connectivity_violation = float("nan")
    if values.shape[1] >= 3:
        volume_violation = values[:, -3]
        roughness_violation = values[:, -2]
        feasible = (volume_violation <= 1e-6) & (roughness_violation <= 1e-6)
        feasible_rate = float(np.mean(feasible))
        mean_volume_violation = float(np.mean(volume_violation))
        mean_roughness_violation = float(np.mean(roughness_violation))
        if np.any(feasible):
            best_feasible_last = float(np.min(last_level[feasible]))
    if values.shape[1] == 4:
        maybe_connectivity_violation = values[:, -4]
        mean_connectivity_violation = float(np.mean(maybe_connectivity_violation))
    return {
        "iteration": float(iteration),
        "eval_count": float(eval_count),
        "best_values": values[0].copy(),
        "best_last": float(last_level[0]),
        "best_feasible_last": best_feasible_last,
        "mean_last": float(np.mean(last_level)),
        "median_last": float(np.median(last_level)),
        "p10_last": float(np.percentile(last_level, 10)),
        "p90_last": float(np.percentile(last_level, 90)),
        "feasible_rate": feasible_rate,
        "mean_volume_violation": mean_volume_violation,
        "mean_roughness_violation": mean_roughness_violation,
        "mean_connectivity_violation": mean_connectivity_violation,
    }


NICHE_HISTORY_COLUMNS = np.asarray(
    [
        "iteration",
        "eval_count",
        "niche_index",
        "capacity",
        "size",
        "best_last",
        "best_feasible_last",
        "mean_last",
        "median_last",
        "p10_last",
        "p90_last",
        "representative_min_hamming",
        "representative_mean_hamming",
    ]
)


def record_niche_history(
    buffer: Any,
    *,
    iteration: int,
    eval_count: int,
) -> np.ndarray:
    """Capture per-niche telemetry when the buffer supports it."""
    if not hasattr(buffer, "get_niche_stats"):
        return np.empty((0, len(NICHE_HISTORY_COLUMNS)), dtype=np.float32)
    rows = []
    for stat in buffer.get_niche_stats():
        rows.append(
            [
                float(iteration),
                float(eval_count),
                stat["niche_index"],
                stat["capacity"],
                stat["size"],
                stat["best_last"],
                stat["best_feasible_last"],
                stat["mean_last"],
                stat["median_last"],
                stat["p10_last"],
                stat["p90_last"],
                stat["representative_min_hamming"],
                stat["representative_mean_hamming"],
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def history_arrays(
    history: list[dict[str, np.ndarray | float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scalar_columns = [
        "iteration",
        "eval_count",
        "best_last",
        "best_feasible_last",
        "mean_last",
        "median_last",
        "p10_last",
        "p90_last",
        "feasible_rate",
        "mean_volume_violation",
        "mean_roughness_violation",
        "mean_connectivity_violation",
    ]
    scalars = np.asarray(
        [[float(row[column]) for column in scalar_columns] for row in history],
        dtype=np.float32,
    )
    best_values = np.asarray([row["best_values"] for row in history], dtype=np.float32)
    return scalars, best_values, np.asarray(scalar_columns)


def live_progress_payload(
    row: dict[str, np.ndarray | float],
    *,
    start_time: float,
    n_iter: int,
) -> dict[str, Any]:
    """Convert a history row to JSON-safe live telemetry."""
    iteration = float(row["iteration"])
    eval_count = float(row["eval_count"])
    elapsed_sec = max(time.perf_counter() - start_time, 1e-9)
    iter_per_sec = iteration / elapsed_sec
    eta_sec = (
        (float(n_iter) - iteration) / iter_per_sec
        if iter_per_sec > 0 and iteration < n_iter
        else 0.0
    )

    def finite_or_none(value: float) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    return {
        "iteration": int(iteration),
        "n_iter": int(n_iter),
        "eval_count": int(eval_count),
        "best_last": finite_or_none(float(row["best_last"])),
        "best_feasible_last": finite_or_none(float(row["best_feasible_last"])),
        "mean_last": finite_or_none(float(row["mean_last"])),
        "median_last": finite_or_none(float(row["median_last"])),
        "p10_last": finite_or_none(float(row["p10_last"])),
        "p90_last": finite_or_none(float(row["p90_last"])),
        "feasible_rate": finite_or_none(float(row["feasible_rate"])),
        "mean_volume_violation": finite_or_none(float(row["mean_volume_violation"])),
        "mean_roughness_violation": finite_or_none(
            float(row["mean_roughness_violation"])
        ),
        "mean_connectivity_violation": finite_or_none(
            float(row["mean_connectivity_violation"])
        ),
        "best_values": [
            finite_or_none(value)
            for value in np.asarray(row["best_values"], dtype=np.float64).ravel()
        ],
        "elapsed_sec": elapsed_sec,
        "iter_per_sec": iter_per_sec,
        "evals_per_sec": eval_count / elapsed_sec,
        "eta_sec": eta_sec,
    }


def plot_buffer_history(
    history: list[dict[str, np.ndarray | float]],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot best/mean buffer behavior for the last value level."""
    if not history:
        return
    scalars, _, columns = history_arrays(history)
    column_index = {str(name): index for index, name in enumerate(columns)}
    evals = scalars[:, column_index["eval_count"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        evals,
        scalars[:, column_index["best_last"]],
        label="best buffer last-level",
        linewidth=2.0,
    )
    ax.plot(
        evals,
        scalars[:, column_index["mean_last"]],
        label="mean buffer last-level",
        linewidth=1.5,
    )
    ax.plot(
        evals,
        scalars[:, column_index["median_last"]],
        label="median buffer last-level",
        linewidth=1.5,
    )
    ax.fill_between(
        evals,
        scalars[:, column_index["p10_last"]],
        scalars[:, column_index["p90_last"]],
        alpha=0.18,
        label="p10-p90 buffer range",
    )
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("buffer value, lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_schedule_decay(decay: str | None) -> str | float | None:
    """Parse optional scheduler cycle decay from CLI text."""
    if decay is None or decay.lower() in {"", "none"}:
        return None
    if decay == "linear":
        return decay
    try:
        return float(decay)
    except ValueError as exc:
        raise ValueError(
            "--curiosity_decay must be none, linear, or a numeric factor"
        ) from exc


def zeropower_via_newton_schulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the zeroth power/orthogonal factor of a 2D gradient."""
    if g.ndim != 2:
        raise ValueError(
            f"Muon zeropower expects a 2D tensor, got shape={tuple(g.shape)}"
        )

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T

    x = x / torch.clamp(x.norm(), min=1e-12)
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x

    if transpose:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Small experimental Muon optimizer for matrix-heavy MLPs.

    2D parameters receive momentum plus Newton-Schulz orthogonalized updates.
    Non-2D parameters fall back to momentum SGD updates.
    """

    def __init__(
        self,
        parameters: Any,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(grad, alpha=1.0 - beta)
                update = grad.add(buf, alpha=beta) if nesterov else buf

                if update.ndim == 2:
                    update_2d = zeropower_via_newton_schulz5(update, steps=ns_steps)
                    scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                    p.add_(update_2d, alpha=-lr * scale)
                else:
                    p.add_(update, alpha=-lr)
        return loss


def make_torch_optimizer(
    optimizer_name: str,
    parameters: Any,
    lr: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=momentum)
    if optimizer_name == "muon":
        return Muon(parameters, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown torch optimizer: {optimizer_name}")


def run_generator_uniformity_warmup(
    gan: components.GAN,
    *,
    steps: int,
    batch_size: int,
    weight: float,
    t: float = 2.0,
) -> None:
    """Update only G with Wang-Isola uniformity before filling the buffer."""
    if steps <= 0:
        return
    if batch_size < 2:
        raise ValueError(
            f"g_uniformity_warmup_batch_size must be >= 2, got {batch_size}"
        )
    if weight < 0:
        raise ValueError(
            f"g_uniformity_warmup_weight must be non-negative, got {weight}"
        )
    if not isinstance(gan.latent_dim, int):
        raise ValueError("G uniformity warmup currently requires int latent_dim")

    logger.info(
        f"Running G-only uniformity warmup: steps={steps} batch_size={batch_size} "
        f"weight={weight:g} t={t:g}"
    )
    for step in range(steps):
        gan.optimizerG.zero_grad()
        z = torch.randn(
            batch_size,
            gan.latent_dim,
            device=gan.device,
            dtype=gan.dtype,
        )
        proposals = gan.G(z)
        loss = weight * uniformity_loss(proposals.reshape(batch_size, -1), t=t)
        loss.backward()
        gan.optimizerG.step()
        if (step + 1) == steps or (step + 1) % max(1, steps // 5) == 0:
            logger.info(
                f"G-only uniformity warmup step {step + 1}/{steps}: loss={loss.item():.6f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GFog FEM cantilever benchmark")
    parser.add_argument(
        "--preset",
        choices=["default", "tom_cantilever_2d"],
        default="default",
        help="Apply a named benchmark preset before constructing the evaluator.",
    )
    parser.add_argument("--grid_width", type=int, default=32)
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--domain_width", type=float, default=1.0)
    parser.add_argument("--domain_height", type=float, default=1.0)
    parser.add_argument("--backend", choices=["scipy", "torchfem"], default="scipy")
    parser.add_argument("--torchfem_src", type=str, default=None)
    parser.add_argument("--torchfem_device", type=str, default="cpu")
    parser.add_argument("--n_iter", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--latent_distribution",
        choices=["normal", "uniform"],
        default="normal",
        help="Latent prior used for fresh z samples and fixed-bank candidates.",
    )
    parser.add_argument("--latent_uniform_low", type=float, default=-1.0)
    parser.add_argument("--latent_uniform_high", type=float, default=1.0)
    parser.add_argument(
        "--encoding",
        choices=[
            "direct",
            "coarse",
            "binary_coarse",
            "topk_volume",
            "sorted_material",
            "coarse_topk_volume",
            "coarse_residual",
            "soft_volume",
            "bar_primitives",
            "tiny_decoder",
        ],
        default="direct",
    )
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--coarse_grid_width", type=int, default=None)
    parser.add_argument("--coarse_grid_height", type=int, default=None)
    parser.add_argument("--tiny_decoder_model", type=str, default="madebyollin/taesd")
    parser.add_argument("--tiny_decoder_latent_channels", type=int, default=4)
    parser.add_argument("--tiny_decoder_latent_height", type=int, default=8)
    parser.add_argument("--tiny_decoder_latent_width", type=int, default=8)
    parser.add_argument("--tiny_decoder_latent_scale", type=float, default=1.0)
    parser.add_argument("--bar_count", type=int, default=16)
    parser.add_argument("--bar_width_min", type=float, default=0.02)
    parser.add_argument("--bar_width_max", type=float, default=0.08)
    parser.add_argument("--bar_edge_softness", type=float, default=0.01)
    parser.add_argument(
        "--sorted_material_profile",
        choices=["binary", "linear", "sigmoid"],
        default="linear",
        help=(
            "Fixed material histogram for --encoding sorted_material. "
            "G controls only the sorted placement/order."
        ),
    )
    parser.add_argument("--sorted_material_steepness", type=float, default=12.0)
    parser.add_argument(
        "--initial_buffer_mode",
        choices=["generator", "random_blobs"],
        default="generator",
        help=(
            "How to fill the initial rank buffer. random_blobs seeds "
            "sorted-material score vectors from diverse blob masks before G training."
        ),
    )
    parser.add_argument("--initial_blob_count_min", type=int, default=1)
    parser.add_argument("--initial_blob_count_max", type=int, default=5)
    parser.add_argument(
        "--initial_blob_radius_min",
        type=float,
        default=0.06,
        help="Minimum normalized blob radius for --initial_buffer_mode random_blobs.",
    )
    parser.add_argument(
        "--initial_blob_radius_max",
        type=float,
        default=0.24,
        help="Maximum normalized blob radius for --initial_buffer_mode random_blobs.",
    )
    parser.add_argument(
        "--initial_blob_score_margin",
        type=float,
        default=1.0,
        help="Score separation between material and void cells in blob seed codes.",
    )
    parser.add_argument(
        "--initial_blob_score_noise",
        type=float,
        default=0.01,
        help="Tie-breaking score noise for blob seed codes.",
    )
    parser.add_argument(
        "--initial_blob_candidate_multiplier",
        type=int,
        default=8,
        help="Over-generate this many blob candidates per requested buffer seed.",
    )
    parser.add_argument(
        "--initial_blob_min_hamming",
        type=float,
        default=0.1,
        help=(
            "Greedy target minimum pairwise Hamming distance between initial "
            "blob seed masks after deduplication."
        ),
    )
    parser.add_argument(
        "--initial_blob_cluster_niches",
        action="store_true",
        help=(
            "For random_blobs plus a niche buffer, initialize each niche from a "
            "balanced Hamming cluster instead of globally diverse seeds."
        ),
    )
    parser.add_argument(
        "--generator_type",
        choices=["mlp", "conv", "set_conv", "set_direct"],
        default="mlp",
    )
    parser.add_argument(
        "--discriminator_type",
        choices=["mlp", "conv", "set_transformer"],
        default="mlp",
    )
    parser.add_argument(
        "--discriminator_spectral_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spectral norm on MLP/Conv discriminator layers.",
    )
    parser.add_argument(
        "--discriminator_activation",
        choices=["leaky_relu", "gelu", "silu"],
        default="leaky_relu",
        help="Activation used by --discriminator_type conv.",
    )
    parser.add_argument("--generator_channels", type=int, default=64)
    parser.add_argument(
        "--generator_output_norm",
        choices=["none", "l2", "centered_l2", "layernorm"],
        default="none",
        help="Normalize each generated genome/score vector before evaluation and D.",
    )
    parser.add_argument("--discriminator_channels", type=int, default=32)
    parser.add_argument("--set_generator_dim", type=int, default=128)
    parser.add_argument("--set_generator_depth", type=int, default=2)
    parser.add_argument("--set_generator_heads", type=int, default=4)
    parser.add_argument("--set_generator_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_generator_dropout", type=float, default=0.0)
    parser.add_argument(
        "--set_generator_elite_context_size",
        type=int,
        default=0,
        help="Number of stochastic top-buffer elite designs exposed to set_conv G.",
    )
    parser.add_argument(
        "--set_generator_elite_context_pool_size",
        type=int,
        default=None,
        help="Sample set_conv G elite context from the top N buffer entries.",
    )
    parser.add_argument("--set_discriminator_dim", type=int, default=128)
    parser.add_argument("--set_discriminator_depth", type=int, default=2)
    parser.add_argument("--set_discriminator_heads", type=int, default=4)
    parser.add_argument("--set_discriminator_mlp_ratio", type=int, default=2)
    parser.add_argument("--set_discriminator_dropout", type=float, default=0.0)
    parser.add_argument(
        "--generator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument(
        "--discriminator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument(
        "--fixed_latent_bank",
        action="store_true",
        help=(
            "Replace fresh Gaussian z samples with a fixed latent bank. "
            "By default the bank size equals buffer_multiplier * batch_size."
        ),
    )
    parser.add_argument("--fixed_latent_bank_size", type=int, default=None)
    parser.add_argument(
        "--fixed_latent_selection",
        choices=["random", "output_diverse", "clustered_niches"],
        default="output_diverse",
        help="How to choose fixed latent bank entries.",
    )
    parser.add_argument(
        "--fixed_latent_candidate_multiplier",
        type=int,
        default=8,
        help="Candidate-pool multiplier for --fixed_latent_selection output_diverse.",
    )
    parser.add_argument(
        "--fixed_latent_chunk_size",
        type=int,
        default=1024,
        help="Chunk size for evaluating candidate latents during fixed-bank selection.",
    )
    parser.add_argument(
        "--fixed_latent_sample_mode",
        choices=["random", "shuffle_cycle", "balanced_niches"],
        default="shuffle_cycle",
        help="How optimizer batches are drawn from the fixed latent bank.",
    )
    parser.add_argument(
        "--fixed_latent_niche_count",
        type=int,
        default=None,
        help=(
            "Number of latent clusters for --fixed_latent_selection clustered_niches. "
            "Defaults to max(niche_buffer_count, 2)."
        ),
    )
    parser.add_argument(
        "--fixed_latent_niche_center_scale",
        type=float,
        default=4.0,
        help="Distance scale of clustered fixed latent niche centers.",
    )
    parser.add_argument(
        "--fixed_latent_niche_within_std",
        type=float,
        default=0.35,
        help="Gaussian spread of fixed latents inside each latent niche.",
    )
    parser.add_argument(
        "--fixed_latent_niche_no_normalize_radius",
        action="store_true",
        help="Do not normalize clustered fixed latents to the usual sqrt(latent_dim) radius.",
    )
    parser.add_argument(
        "--fixed_latent_noise_std",
        type=float,
        default=0.0,
        help="Sample z around fixed bank means with this Gaussian stddev.",
    )
    parser.add_argument(
        "--fixed_latent_noise_no_normalize",
        action="store_true",
        help="Do not divide jittered fixed latents by sqrt(1 + noise_std^2).",
    )
    parser.add_argument(
        "--fixed_latent_uniformity_weight",
        type=float,
        default=0.0,
        help="Extra G-side uniformity weight on G(z) sampled from the fixed latent bank.",
    )
    parser.add_argument(
        "--fixed_latent_uniformity_batch_size",
        type=int,
        default=128,
        help="Number of fixed-bank latents used by the extra bank uniformity loss.",
    )
    parser.add_argument("--fixed_latent_uniformity_t", type=float, default=2.0)
    parser.add_argument(
        "--fixed_latent_uniformity_sample_mode",
        choices=["random", "shuffle_cycle"],
        default="shuffle_cycle",
    )
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument(
        "--buffer_diversity_min_hamming",
        type=float,
        default=0.0,
        help=(
            "If >0, rebuild the elite buffer as a ranked Hamming-diverse set. "
            "Unfilled slots fall back to rank order."
        ),
    )
    parser.add_argument(
        "--buffer_diversity_topk_frac",
        type=float,
        default=0.48,
        help="Top-k fraction used to binarize raw scores for buffer diversity.",
    )
    parser.add_argument(
        "--niche_buffer_count",
        type=int,
        default=1,
        help=(
            "If >1, split the elite archive into this many independent design "
            "niches instead of one global buffer."
        ),
    )
    parser.add_argument(
        "--niche_buffer_min_hamming",
        type=float,
        default=0.1,
        help=(
            "Target minimum decoded-design Hamming distance between niche "
            "representatives."
        ),
    )
    parser.add_argument(
        "--niche_buffer_view_mode",
        choices=["balanced", "global"],
        default="balanced",
        help=(
            "Expose either a round-robin balanced view across niches or a "
            "globally sorted view to D/history."
        ),
    )
    parser.add_argument(
        "--niche_buffer_cross_min_hamming",
        type=float,
        default=0.0,
        help=(
            "If >0, reject inserts whose decoded design is closer than this "
            "Hamming distance to top designs in another niche."
        ),
    )
    parser.add_argument(
        "--niche_buffer_cross_reference_top_k",
        type=int,
        default=1,
        help=(
            "Number of top designs per other niche used by "
            "--niche_buffer_cross_min_hamming."
        ),
    )
    parser.add_argument(
        "--niche_output_separation_weight",
        type=float,
        default=0.0,
        help=(
            "Extra G-side loss weight that repels raw output prototypes for "
            "samples routed to different stable niche anchors."
        ),
    )
    parser.add_argument(
        "--niche_output_separation_margin",
        type=float,
        default=0.35,
        help=(
            "Minimum centered-L2 distance between generated output prototypes "
            "assigned to different niches."
        ),
    )
    parser.add_argument("--curiosity", type=float, default=10.0)
    parser.add_argument(
        "--curiosity_space",
        choices=["raw", "topology", "plummer"],
        default="raw",
        help=(
            "Apply curiosity to raw outputs, decoded topology fields, or a "
            "Plummer repulsion kernel in genome space."
        ),
    )
    parser.add_argument(
        "--plummer_power",
        type=float,
        default=1.0,
        help="Inverse-power exponent for --curiosity_space plummer.",
    )
    parser.add_argument(
        "--plummer_eps",
        type=float,
        default=1e-3,
        help="Softening constant added to mean squared distances for Plummer repulsion.",
    )
    parser.add_argument(
        "--plummer_normalize",
        choices=["none", "layernorm", "l2"],
        default="layernorm",
        help="Per-sample normalization before Plummer distances.",
    )
    parser.add_argument(
        "--plummer_terms",
        choices=["batch", "buffer", "batch_buffer"],
        default="batch_buffer",
        help=(
            "Which Plummer repulsion terms to apply: generated batch only, "
            "generated-vs-buffer only, or both."
        ),
    )
    parser.add_argument(
        "--curiosity_schedule",
        choices=["none", "warmup_cosine", "warmup_cosine_annealing", "cosine_ramp"],
        default="none",
        help="Optional schedule multiplier for curiosity weight.",
    )
    parser.add_argument("--curiosity_warmup_frac", type=float, default=0.05)
    parser.add_argument("--curiosity_min", type=float, default=0.0)
    parser.add_argument(
        "--curiosity_cycles",
        type=int,
        default=4,
        help="Number of cycles for --curiosity_schedule warmup_cosine_annealing.",
    )
    parser.add_argument(
        "--curiosity_decay",
        type=str,
        default=None,
        help=(
            "Cycle peak decay for warmup_cosine_annealing: none, linear, "
            "or a numeric factor such as 0.8."
        ),
    )
    parser.add_argument(
        "--g_uniformity_warmup_steps",
        type=int,
        default=0,
        help="Run this many G-only uniformity updates before initial buffer filling.",
    )
    parser.add_argument(
        "--g_uniformity_warmup_batch_size",
        type=int,
        default=None,
        help="Batch size for G-only uniformity warmup. Defaults to --batch_size.",
    )
    parser.add_argument(
        "--g_uniformity_warmup_weight",
        type=float,
        default=None,
        help=(
            "Weight for G-only uniformity warmup. Defaults to --curiosity, "
            "or 1 if curiosity is 0."
        ),
    )
    parser.add_argument(
        "--g_uniformity_warmup_t",
        type=float,
        default=2.0,
        help="Wang-Isola t parameter for G-only uniformity warmup.",
    )
    parser.add_argument(
        "--curiosity_reference",
        choices=["buffer", "batch"],
        default="buffer",
        help="Apply curiosity to generated batch only or generated batch plus buffer.",
    )
    parser.add_argument(
        "--train_on_decoded",
        action="store_true",
        help=(
            "Train discriminator/curiosity on decoded physical density fields instead "
            "of raw generator codes. Supports direct, soft_volume, coarse, and coarse_residual encodings."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer_type",
        choices=[
            "default",
            "hinge",
            "lsgan",
            "wgan",
            "wgangp",
            "plackett_luce",
            "buffer_plackett_luce",
            "contextual_plackett_luce",
            "calibrated_utility",
            "hybrid_contextual_utility",
            "ranked_lsgan",
            "ranked_default",
            "ranked_wgan",
            "quantile_ranked_default",
            "quantile_ranked_value_default",
        ],
        default="default",
    )
    parser.add_argument("--ranker_list_size", type=int, default=32)
    parser.add_argument("--ranker_steps", type=int, default=1)
    parser.add_argument("--ranker_archive_size", type=int, default=8192)
    parser.add_argument("--ranker_generator_elite_margin", action="store_true")
    parser.add_argument(
        "--ranker_sample_pool_size",
        type=int,
        default=None,
        help="Sample ranker lists from only the top N buffer entries.",
    )
    parser.add_argument(
        "--ranker_sample_mode",
        choices=["random_top_pool", "top_k"],
        default="random_top_pool",
        help="Use random sorted samples from the ranker pool or exact top-k lists.",
    )
    parser.add_argument(
        "--ranker_weight",
        type=float,
        default=0.1,
        help="Weight for auxiliary PL rank loss in ranked_* optimizers.",
    )
    parser.add_argument(
        "--ranker_target_curve",
        choices=["linear", "exp"],
        default="linear",
        help="Rank target curve for quantile_ranked_default.",
    )
    parser.add_argument(
        "--ranker_tau",
        type=float,
        default=16.0,
        help="Exponential rank target decay for --ranker_target_curve exp.",
    )
    parser.add_argument(
        "--ranker_target_scope",
        choices=["local", "global"],
        default="local",
        help="Use local sampled-list ranks or global buffer ranks for quantile targets.",
    )
    parser.add_argument(
        "--ranker_niche_local_targets",
        action="store_true",
        help=(
            "For quantile_ranked_default with a niche buffer, build rank targets "
            "separately inside each niche before averaging the D real loss."
        ),
    )
    parser.add_argument(
        "--ranker_list_repeats",
        type=int,
        default=1,
        help="Average this many independent rank lists inside one D update.",
    )
    parser.add_argument(
        "--ranker_fake_weight",
        type=float,
        default=1.0,
        help="Weight for the quantile-ranker D(fake)=0 term.",
    )
    parser.add_argument(
        "--ranker_fake_repeats",
        type=int,
        default=1,
        help="Average this many generated-fake batches inside one D update.",
    )
    parser.add_argument(
        "--d_score_center_weight",
        type=float,
        default=0.0,
        help="Penalty weight for centering contextual PL D scores around zero.",
    )
    parser.add_argument(
        "--d_score_scale_weight",
        type=float,
        default=0.0,
        help="Penalty weight for matching contextual PL D score std.",
    )
    parser.add_argument(
        "--d_score_target_std",
        type=float,
        default=1.0,
        help="Target standard deviation for contextual PL D score scale.",
    )
    parser.add_argument(
        "--utility_target_scale",
        type=float,
        default=100.0,
        help="Scale for calibrated utility target: utility = -last_objective / scale.",
    )
    parser.add_argument(
        "--utility_loss",
        choices=["smooth_l1", "mse"],
        default="smooth_l1",
        help="Regression loss for --optimizer_type calibrated_utility.",
    )
    parser.add_argument(
        "--utility_weight",
        type=float,
        default=0.1,
        help="D-side utility calibration weight for hybrid contextual utility.",
    )
    parser.add_argument(
        "--generator_utility_weight",
        type=float,
        default=0.0,
        help=(
            "G-side utility score maximization weight for hybrid contextual utility. "
            "Must stay 0 for quantile_ranked_value_default."
        ),
    )
    parser.add_argument(
        "--utility_clip",
        type=float,
        default=3.0,
        help="Clip local utility targets for hybrid contextual utility.",
    )
    parser.add_argument(
        "--proposal_pool_size",
        type=int,
        default=None,
        help="Generate this many proposals, score by D, then select the final batch.",
    )
    parser.add_argument(
        "--proposal_top_k",
        type=int,
        default=None,
        help="Prefilter proposal pool to top-k by D before diversity selection.",
    )
    parser.add_argument(
        "--proposal_diversity_min_hamming",
        type=float,
        default=0.0,
        help="Greedy Hamming-distance threshold for selected proposal batch.",
    )
    parser.add_argument(
        "--proposal_diversity_topk_frac",
        type=float,
        default=0.48,
        help="Top-k fraction used to binarize raw scores for proposal diversity.",
    )
    parser.add_argument(
        "--proposal_buffer_novelty_min_hamming",
        type=float,
        default=0.0,
        help=(
            "Reject D-ranked proposal candidates whose decoded binary design is "
            "closer than this Hamming distance to any top buffer reference."
        ),
    )
    parser.add_argument(
        "--proposal_buffer_novelty_reference_size",
        type=int,
        default=128,
        help="Number of top buffer designs used as duplicate references.",
    )
    parser.add_argument(
        "--proposal_buffer_reject_exact_design_duplicates",
        action="store_true",
        help=(
            "Reject proposal candidates only when their decoded binary design "
            "exactly matches a top buffer reference."
        ),
    )
    parser.add_argument(
        "--proposal_buffer_novelty_threshold",
        type=float,
        default=0.5,
        help="Density threshold used to binarize decoded designs for novelty checks.",
    )
    parser.add_argument(
        "--proposal_evolution_fraction",
        type=float,
        default=0.0,
        help="Fraction of the D-ranked proposal pool replaced by evolved children.",
    )
    parser.add_argument(
        "--proposal_evolution_parent_source",
        choices=["pool", "pool_buffer"],
        default="pool",
        help="Use only generated pool parents or mix generated proposals with buffer elites.",
    )
    parser.add_argument(
        "--proposal_evolution_crossover",
        choices=["uniform", "row", "rect"],
        default="uniform",
        help="Crossover operator for proposal-pool evolution.",
    )
    parser.add_argument(
        "--proposal_evolution_mutation_rate",
        type=float,
        default=0.01,
        help="Per-coordinate mutation probability for evolved proposal-pool children.",
    )
    parser.add_argument(
        "--proposal_evolution_mutation_scale",
        type=float,
        default=0.10,
        help="Gaussian mutation stddev for evolved proposal-pool children.",
    )
    parser.add_argument(
        "--proposal_gradient_steps",
        type=int,
        default=0,
        help="D-gradient ascent steps applied to proposal score maps before FEM.",
    )
    parser.add_argument(
        "--proposal_gradient_step_size",
        type=float,
        default=0.05,
        help=(
            "Step size for continuous D-gradient refinement, or material fraction "
            "to swap for --proposal_gradient_mode swap."
        ),
    )
    parser.add_argument(
        "--proposal_gradient_mode",
        choices=["continuous", "swap"],
        default="continuous",
        help="Use continuous score ascent or top-k-preserving promote/demote swaps.",
    )
    parser.add_argument(
        "--proposal_gradient_no_normalize",
        action="store_true",
        help="Use raw D gradients instead of per-sample normalized gradients.",
    )
    parser.add_argument(
        "--proposal_gradient_noise",
        type=float,
        default=0.0,
        help="Gaussian noise added after each D-gradient proposal refinement step.",
    )
    parser.add_argument(
        "--proposal_gradient_keep_original",
        action="store_true",
        help="Rank both original and D-gradient-refined proposals before selection.",
    )
    parser.add_argument(
        "--ga_offspring_fraction",
        type=float,
        default=0.0,
        help="Fraction of each evaluated ranker batch replaced by GA children.",
    )
    parser.add_argument(
        "--ga_pool_size",
        type=int,
        default=None,
        help="Number of GA children generated before D preselection.",
    )
    parser.add_argument(
        "--ga_parent_pool_size",
        type=int,
        default=128,
        help="Sample crossover parents from the top N buffer entries.",
    )
    parser.add_argument(
        "--ga_mutation_rate",
        type=float,
        default=0.02,
        help="Per-coordinate Gaussian mutation probability for GA children.",
    )
    parser.add_argument(
        "--ga_mutation_scale",
        type=float,
        default=0.25,
        help="Gaussian mutation stddev for GA children.",
    )
    parser.add_argument(
        "--g_torch_optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop", "muon"],
        default="adam",
    )
    parser.add_argument(
        "--d_torch_optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop", "muon"],
        default="adam",
    )
    parser.add_argument("--g_lr", type=float, default=0.01)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument("--g_momentum", type=float, default=0.9)
    parser.add_argument("--d_momentum", type=float, default=0.9)
    parser.add_argument("--discriminator_steps", type=int, default=3)
    parser.add_argument(
        "--elite_sampling",
        choices=["random_top_k", "top_k"],
        default="random_top_k",
        help="How GAN optimizers sample real elite batches from the buffer.",
    )
    parser.add_argument(
        "--elite_pool_size",
        type=int,
        default=None,
        help="Top-k pool size for random_top_k elite sampling. Defaults to buffer size.",
    )
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--roughness_max", type=float, default=0.18)
    parser.add_argument(
        "--connectivity_max",
        type=float,
        default=None,
        help=(
            "Optional max disconnected solid fraction. When set, f adds a "
            "connectivity violation objective before volume/roughness/compliance."
        ),
    )
    parser.add_argument("--volume_ladder", nargs="*", type=float, default=[])
    parser.add_argument("--compliance_ladder", nargs="*", type=float, default=[])
    parser.add_argument("--roughness_ladder", nargs="*", type=float, default=[])
    parser.add_argument("--connectivity_ladder", nargs="*", type=float, default=[])
    parser.add_argument(
        "--diversity_ladder",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Minimum boundary-Chamfer distance from current top-buffer designs. "
            "Adds max(bound - chamfer, 0) violations; usually prefer "
            "--ladder_sequence for interleaving with compliance."
        ),
    )
    parser.add_argument(
        "--diversity_reference_size",
        type=int,
        default=32,
        help="Number of top-buffer designs used as Chamfer diversity references.",
    )
    parser.add_argument(
        "--diversity_chamfer_max_points",
        type=int,
        default=256,
        help="Maximum boundary points per design for the Chamfer diversity proxy.",
    )
    parser.add_argument(
        "--removal_ladder_volumes",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Optional staged material-removal objective. G emits element "
            "importance scores; f keeps top-k material at each listed volume."
        ),
    )
    parser.add_argument(
        "--removal_ladder_compliances",
        nargs="*",
        type=float,
        default=[],
        help=(
            "Compliance thresholds for --removal_ladder_volumes. When set, f "
            "returns per-stage violations plus final compliance tie-break."
        ),
    )
    parser.add_argument(
        "--removal_ladder_connectivity_max",
        type=float,
        default=None,
        help=(
            "Optional max disconnected solid fraction for every removal-ladder "
            "stage. A value of 0 requires all kept material to be connected to "
            "the left support before the stage compliance violation is compared."
        ),
    )
    parser.add_argument(
        "--ladder_sequence",
        nargs="*",
        type=str,
        default=[],
        help="Explicit interleaved ladder sequence like volume:0.52 compliance:318 volume:0.50 compliance:314",
    )
    parser.add_argument(
        "--levels_ladder",
        nargs="*",
        type=str,
        default=[],
        help=(
            "Use official Levels.ladder with raw objectives, e.g. "
            "volume:0.55,0.50 compliance:130,110,100"
        ),
    )
    parser.add_argument(
        "--levels_ladder_final_open",
        type=str,
        default="compliance",
        help="Optional objective name for final open rung tie-breaker.",
    )
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--e_max", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.3)
    parser.add_argument("--load_scale", type=float, default=1.0)
    parser.add_argument(
        "--load_case",
        choices=LOAD_CASE_CHOICES,
        default="center_point",
    )
    parser.add_argument(
        "--robust_load_cases",
        nargs="*",
        choices=LOAD_CASE_CHOICES,
        default=[],
        help=(
            "Evaluate each design under these load cases and use the aggregate "
            "compliance as f's compliance objective. Empty keeps --load_case only."
        ),
    )
    parser.add_argument(
        "--robust_load_aggregate",
        choices=["max", "mean", "cvar"],
        default="max",
        help="How to aggregate compliances across --robust_load_cases.",
    )
    parser.add_argument(
        "--robust_load_cvar_frac",
        type=float,
        default=0.5,
        help="Worst-case fraction used when --robust_load_aggregate cvar.",
    )
    parser.add_argument(
        "--fem_workers",
        type=int,
        default=1,
        help="Number of CPU worker threads for parallel SciPy FEM batch evaluation.",
    )
    parser.add_argument(
        "--compliance_solver",
        choices=["direct", "matrix_free_cg"],
        default="direct",
        help="Forward compliance solver used by the SciPy backend.",
    )
    parser.add_argument(
        "--matrix_free_cg_max_iter",
        type=int,
        default=1000,
        help="Maximum Jacobi-CG iterations for --compliance_solver matrix_free_cg.",
    )
    parser.add_argument(
        "--matrix_free_cg_tol",
        type=float,
        default=1e-6,
        help="Relative residual tolerance for --compliance_solver matrix_free_cg.",
    )
    parser.add_argument(
        "--matrix_free_cg_device",
        type=str,
        default="cpu",
        help="Torch device for the matrix-free CG solve.",
    )
    parser.add_argument(
        "--matrix_free_cg_dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Torch dtype for the matrix-free CG solve. Use float32 for MPS.",
    )
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument(
        "--density_filter_warmup_iters",
        type=int,
        default=0,
        help=(
            "If >0, use --density_filter_radius for this many optimizer iterations, "
            "then switch to --density_filter_final_radius and re-score the buffer."
        ),
    )
    parser.add_argument(
        "--density_filter_final_radius",
        type=int,
        default=0,
        help=(
            "Density filter radius after --density_filter_warmup_iters. "
            "Use 0 to remove smoothing after the warmup."
        ),
    )
    parser.add_argument("--projection_beta", type=float, default=0.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument(
        "--binhead_connect_support",
        action="store_true",
        help=(
            "After binary top-k/sorted decoding, keep support-connected material "
            "and refill removed cells near that component before FEM."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/fem_cantilever"),
    )
    parser.add_argument(
        "--history_interval",
        type=int,
        default=1,
        help="Record buffer telemetry every N optimizer iterations.",
    )
    parser.add_argument(
        "--design_history_top_k",
        type=int,
        default=9,
        help=(
            "Store this many decoded top-buffer designs at each history checkpoint. "
            "Use 0 to disable design-history artifacts."
        ),
    )
    parser.add_argument(
        "--no_live_progress",
        action="store_true",
        help=(
            "Disable append-only live_progress_*.jsonl telemetry and periodic "
            "progress log lines during optimization."
        ),
    )
    parser.add_argument(
        "--no_history_plot",
        action="store_true",
        help="Store history arrays but skip the PNG history plot.",
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.preset == "tom_cantilever_2d":
        args.grid_width = 150
        args.grid_height = 100
        args.domain_width = 1.5
        args.domain_height = 1.0
        args.volume_max = 0.48
        args.e_max = 196.0
        args.e_min = 196.0e-6
        args.poisson_ratio = 0.3
        args.load_scale = 1.0
        args.load_case = "tom_two_patches"
        if args.density_filter_radius == 1:
            args.density_filter_radius = 0

    if args.density_filter_radius < 0:
        raise ValueError(
            f"--density_filter_radius must be non-negative, got {args.density_filter_radius}"
        )
    if args.density_filter_warmup_iters < 0:
        raise ValueError(
            "--density_filter_warmup_iters must be non-negative, got "
            f"{args.density_filter_warmup_iters}"
        )
    if args.density_filter_final_radius < 0:
        raise ValueError(
            "--density_filter_final_radius must be non-negative, got "
            f"{args.density_filter_final_radius}"
        )

    if args.train_on_decoded:
        if args.encoding not in {"direct", "soft_volume", "coarse", "coarse_residual"}:
            raise ValueError(
                "--train_on_decoded only supports direct, soft_volume, coarse, and coarse_residual encodings"
            )
        if args.hard_binarize:
            raise ValueError("--train_on_decoded does not support --hard_binarize")
    if args.encoding == "tiny_decoder":
        if args.generator_type != "mlp":
            raise ValueError(
                "--encoding tiny_decoder currently requires --generator_type mlp"
            )
        if args.curiosity_space == "topology":
            raise ValueError(
                "--encoding tiny_decoder does not support --curiosity_space topology yet"
            )
    if args.encoding == "bar_primitives":
        if args.bar_count <= 0:
            raise ValueError(f"--bar_count must be positive, got {args.bar_count}")
        if args.bar_width_min <= 0:
            raise ValueError(
                f"--bar_width_min must be positive, got {args.bar_width_min}"
            )
        if args.bar_width_max < args.bar_width_min:
            raise ValueError(
                "--bar_width_max must be >= --bar_width_min, got "
                f"{args.bar_width_max} < {args.bar_width_min}"
            )
        if args.bar_edge_softness <= 0:
            raise ValueError(
                f"--bar_edge_softness must be positive, got {args.bar_edge_softness}"
            )
    if args.fem_workers < 1:
        raise ValueError(f"--fem_workers must be >= 1, got {args.fem_workers}")
    if args.backend == "torchfem" and args.fem_workers != 1:
        raise ValueError("--fem_workers > 1 is only supported for --backend scipy")
    if args.backend != "scipy" and args.compliance_solver != "direct":
        raise ValueError("--compliance_solver matrix_free_cg requires --backend scipy")
    if args.matrix_free_cg_max_iter < 1:
        raise ValueError(
            "--matrix_free_cg_max_iter must be >= 1, got "
            f"{args.matrix_free_cg_max_iter}"
        )
    if args.matrix_free_cg_tol <= 0.0:
        raise ValueError(
            f"--matrix_free_cg_tol must be positive, got {args.matrix_free_cg_tol}"
        )
    if (
        args.compliance_solver == "matrix_free_cg"
        and args.matrix_free_cg_device.startswith("mps")
        and args.matrix_free_cg_dtype == "float64"
    ):
        raise ValueError(
            "--matrix_free_cg_device mps requires --matrix_free_cg_dtype float32"
        )
    if args.robust_load_cvar_frac <= 0.0 or args.robust_load_cvar_frac > 1.0:
        raise ValueError(
            "--robust_load_cvar_frac must be in (0, 1], got "
            f"{args.robust_load_cvar_frac}"
        )
    if args.backend == "torchfem" and args.robust_load_cases:
        raise ValueError("--robust_load_cases is only supported for --backend scipy")
    if args.compliance_solver == "matrix_free_cg" and args.robust_load_cases:
        raise ValueError(
            "--compliance_solver matrix_free_cg currently supports one load case"
        )
    if args.removal_ladder_volumes:
        if args.encoding not in {"topk_volume", "sorted_material"}:
            raise ValueError(
                "--removal_ladder_volumes currently requires --encoding "
                "topk_volume or sorted_material so G emits one score per element"
            )
        for volume in args.removal_ladder_volumes:
            if volume <= 0.0 or volume > 1.0:
                raise ValueError(
                    f"removal ladder volumes must be in (0, 1], got {volume}"
                )
        if args.removal_ladder_compliances and (
            len(args.removal_ladder_compliances) != len(args.removal_ladder_volumes)
        ):
            raise ValueError(
                "--removal_ladder_compliances must have the same length as "
                "--removal_ladder_volumes"
            )
        if args.removal_ladder_connectivity_max is not None and (
            args.removal_ladder_connectivity_max < 0.0
            or args.removal_ladder_connectivity_max > 1.0
        ):
            raise ValueError(
                "--removal_ladder_connectivity_max must be in [0, 1], got "
                f"{args.removal_ladder_connectivity_max}"
            )
        if (
            args.volume_ladder
            or args.compliance_ladder
            or args.roughness_ladder
            or args.connectivity_ladder
            or args.diversity_ladder
            or args.ladder_sequence
            or args.levels_ladder
        ):
            raise ValueError(
                "--removal_ladder_volumes cannot be combined with other ladder args"
            )
    if (
        args.curiosity_space == "plummer"
        and args.plummer_terms == "buffer"
        and args.curiosity_reference != "buffer"
    ):
        raise ValueError("--plummer_terms buffer requires --curiosity_reference buffer")
    if args.diversity_reference_size < 1:
        raise ValueError(
            "--diversity_reference_size must be >= 1, got "
            f"{args.diversity_reference_size}"
        )
    if args.diversity_chamfer_max_points < 1:
        raise ValueError(
            "--diversity_chamfer_max_points must be >= 1, got "
            f"{args.diversity_chamfer_max_points}"
        )
    if args.initial_buffer_mode == "random_blobs":
        if args.encoding != "sorted_material":
            raise ValueError(
                "--initial_buffer_mode random_blobs currently requires "
                "--encoding sorted_material"
            )
        if args.initial_blob_count_min < 1:
            raise ValueError(
                "--initial_blob_count_min must be >= 1, got "
                f"{args.initial_blob_count_min}"
            )
        if args.initial_blob_count_max < args.initial_blob_count_min:
            raise ValueError(
                "--initial_blob_count_max must be >= --initial_blob_count_min, got "
                f"{args.initial_blob_count_max} and {args.initial_blob_count_min}"
            )
        if not 0.0 < args.initial_blob_radius_min <= args.initial_blob_radius_max:
            raise ValueError(
                "--initial_blob_radius_min/max must satisfy 0 < min <= max, got "
                f"{args.initial_blob_radius_min} and {args.initial_blob_radius_max}"
            )
        if args.initial_blob_score_margin <= 0:
            raise ValueError(
                "--initial_blob_score_margin must be positive, got "
                f"{args.initial_blob_score_margin}"
            )
        if args.initial_blob_score_noise < 0:
            raise ValueError(
                "--initial_blob_score_noise must be non-negative, got "
                f"{args.initial_blob_score_noise}"
            )
        if args.initial_blob_candidate_multiplier < 1:
            raise ValueError(
                "--initial_blob_candidate_multiplier must be >= 1, got "
                f"{args.initial_blob_candidate_multiplier}"
            )
        if not 0.0 <= args.initial_blob_min_hamming <= 1.0:
            raise ValueError(
                "--initial_blob_min_hamming must be in [0, 1], got "
                f"{args.initial_blob_min_hamming}"
            )
    elif args.initial_blob_cluster_niches:
        raise ValueError(
            "--initial_blob_cluster_niches requires --initial_buffer_mode random_blobs"
        )
    if args.history_interval < 1:
        raise ValueError(
            f"--history_interval must be >= 1, got {args.history_interval}"
        )
    if args.design_history_top_k < 0:
        raise ValueError(
            f"--design_history_top_k must be non-negative, got {args.design_history_top_k}"
        )
    if args.proposal_buffer_novelty_min_hamming < 0:
        raise ValueError(
            "--proposal_buffer_novelty_min_hamming must be non-negative, got "
            f"{args.proposal_buffer_novelty_min_hamming}"
        )
    if args.proposal_buffer_novelty_reference_size < 1:
        raise ValueError(
            "--proposal_buffer_novelty_reference_size must be >= 1, got "
            f"{args.proposal_buffer_novelty_reference_size}"
        )
    if not 0.0 <= args.proposal_buffer_novelty_threshold <= 1.0:
        raise ValueError(
            "--proposal_buffer_novelty_threshold must be in [0, 1], got "
            f"{args.proposal_buffer_novelty_threshold}"
        )
    if args.niche_buffer_count < 1:
        raise ValueError(
            f"--niche_buffer_count must be >= 1, got {args.niche_buffer_count}"
        )
    if not 0.0 <= args.niche_buffer_min_hamming <= 1.0:
        raise ValueError(
            "--niche_buffer_min_hamming must be in [0, 1], got "
            f"{args.niche_buffer_min_hamming}"
        )
    if args.niche_buffer_count > args.buffer_multiplier * args.batch_size:
        raise ValueError(
            "--niche_buffer_count must not exceed total buffer size, got "
            f"{args.niche_buffer_count} niches for "
            f"{args.buffer_multiplier * args.batch_size} buffer slots"
        )
    if args.niche_buffer_count > 1 and args.buffer_diversity_min_hamming > 0:
        raise ValueError(
            "--niche_buffer_count cannot currently be combined with "
            "--buffer_diversity_min_hamming"
        )
    if not 0.0 <= args.niche_buffer_cross_min_hamming <= 1.0:
        raise ValueError(
            "--niche_buffer_cross_min_hamming must be in [0, 1], got "
            f"{args.niche_buffer_cross_min_hamming}"
        )
    if args.niche_buffer_cross_reference_top_k <= 0:
        raise ValueError(
            "--niche_buffer_cross_reference_top_k must be positive, got "
            f"{args.niche_buffer_cross_reference_top_k}"
        )
    if args.niche_buffer_cross_min_hamming > 0 and args.niche_buffer_count <= 1:
        raise ValueError(
            "--niche_buffer_cross_min_hamming requires --niche_buffer_count > 1"
        )
    if args.niche_output_separation_weight < 0:
        raise ValueError(
            "--niche_output_separation_weight must be non-negative, got "
            f"{args.niche_output_separation_weight}"
        )
    if args.niche_output_separation_margin <= 0:
        raise ValueError(
            "--niche_output_separation_margin must be positive, got "
            f"{args.niche_output_separation_margin}"
        )
    if args.niche_output_separation_weight > 0 and args.niche_buffer_count <= 1:
        raise ValueError(
            "--niche_output_separation_weight requires --niche_buffer_count > 1"
        )
    if args.initial_blob_cluster_niches and args.niche_buffer_count <= 1:
        raise ValueError(
            "--initial_blob_cluster_niches requires --niche_buffer_count > 1"
        )
    if args.ranker_niche_local_targets:
        if args.optimizer_type != "quantile_ranked_default":
            raise ValueError(
                "--ranker_niche_local_targets currently requires "
                "--optimizer_type quantile_ranked_default"
            )
        if args.niche_buffer_count <= 1:
            raise ValueError(
                "--ranker_niche_local_targets requires --niche_buffer_count > 1"
            )
        if args.ranker_list_size < args.niche_buffer_count:
            raise ValueError(
                "--ranker_list_size must be >= --niche_buffer_count when "
                "--ranker_niche_local_targets is set"
            )

    ladder_sequence = parse_ladder_sequence(args.ladder_sequence)
    levels_ladder_rungs = parse_levels_ladder_specs(args.levels_ladder)
    if levels_ladder_rungs and (
        args.volume_ladder
        or args.compliance_ladder
        or args.roughness_ladder
        or args.connectivity_ladder
        or args.diversity_ladder
        or ladder_sequence
        or args.connectivity_max is not None
    ):
        raise ValueError(
            "--levels_ladder cannot be combined with topology-specific ladder/connectivity args"
        )

    cfg = FEMConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        coarse_grid_width=args.coarse_grid_width,
        coarse_grid_height=args.coarse_grid_height,
        encoding=args.encoding,
        residual_scale=args.residual_scale,
        simp_p=args.simp_p,
        e_min=args.e_min,
        e_max=args.e_max,
        poisson_ratio=args.poisson_ratio,
        volume_max=args.volume_max,
        roughness_max=args.roughness_max,
        connectivity_max=args.connectivity_max,
        volume_ladder=tuple(args.volume_ladder),
        compliance_ladder=tuple(args.compliance_ladder),
        roughness_ladder=tuple(args.roughness_ladder),
        connectivity_ladder=tuple(args.connectivity_ladder),
        diversity_ladder=tuple(args.diversity_ladder),
        ladder_sequence=ladder_sequence,
        use_levels_ladder=bool(levels_ladder_rungs),
        levels_ladder_objectives=tuple(rung.name for rung in levels_ladder_rungs),
        diversity_reference_size=args.diversity_reference_size,
        diversity_chamfer_max_points=args.diversity_chamfer_max_points,
        load_scale=args.load_scale,
        load_case=args.load_case,
        robust_load_cases=tuple(args.robust_load_cases),
        robust_load_aggregate=args.robust_load_aggregate,
        robust_load_cvar_frac=args.robust_load_cvar_frac,
        removal_ladder_volumes=tuple(args.removal_ladder_volumes),
        removal_ladder_compliances=tuple(args.removal_ladder_compliances),
        removal_ladder_connectivity_max=args.removal_ladder_connectivity_max,
        fem_workers=args.fem_workers,
        compliance_solver=args.compliance_solver,
        matrix_free_cg_max_iter=args.matrix_free_cg_max_iter,
        matrix_free_cg_tol=args.matrix_free_cg_tol,
        matrix_free_cg_device=args.matrix_free_cg_device,
        matrix_free_cg_dtype=args.matrix_free_cg_dtype,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
        binhead_connect_support=args.binhead_connect_support,
        tiny_decoder_model=args.tiny_decoder_model,
        tiny_decoder_latent_channels=args.tiny_decoder_latent_channels,
        tiny_decoder_latent_height=args.tiny_decoder_latent_height,
        tiny_decoder_latent_width=args.tiny_decoder_latent_width,
        tiny_decoder_latent_scale=args.tiny_decoder_latent_scale,
        bar_count=args.bar_count,
        bar_width_min=args.bar_width_min,
        bar_width_max=args.bar_width_max,
        bar_edge_softness=args.bar_edge_softness,
        sorted_material_profile=args.sorted_material_profile,
        sorted_material_steepness=args.sorted_material_steepness,
    )
    if args.backend == "scipy":
        evaluator = FEMCantileverEvaluator(cfg)
    elif args.backend == "torchfem":
        evaluator = TorchFEMCantileverEvaluator(
            cfg,
            torchfem_src=args.torchfem_src,
            device=args.torchfem_device,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    code_height = args.coarse_grid_height or args.grid_height
    code_width = args.coarse_grid_width or args.grid_width
    code_channels = 1
    if args.encoding == "tiny_decoder":
        code_height = args.tiny_decoder_latent_height
        code_width = args.tiny_decoder_latent_width
        code_channels = args.tiny_decoder_latent_channels
    elif args.encoding == "bar_primitives":
        code_height = 1
        code_width = 5 * args.bar_count
    elif args.encoding in {"direct", "soft_volume", "topk_volume", "sorted_material"}:
        code_height = args.grid_height
        code_width = args.grid_width
    coarse_dim = code_channels * code_height * code_width
    full_dim = args.grid_height * args.grid_width
    f_dim = coarse_dim + full_dim if args.encoding == "coarse_residual" else coarse_dim
    d_input_dim = full_dim if args.train_on_decoded else f_dim
    device = torch.device("cpu")
    proposal_design_proxy: DesignProxyFn | None = None
    if (
        args.proposal_buffer_novelty_min_hamming > 0
        or args.proposal_buffer_reject_exact_design_duplicates
        or args.niche_buffer_count > 1
    ):
        novelty_threshold = args.proposal_buffer_novelty_threshold

        def proposal_design_proxy(candidates: torch.Tensor) -> torch.Tensor:
            designs = evaluator.decode_designs_numpy(candidates.detach().cpu().numpy())
            return torch.from_numpy(designs >= novelty_threshold)

    if args.generator_type == "mlp":
        g = MLP(
            input_dim=args.latent_dim,
            output_dim=f_dim,
            hidden_dims=args.generator_hidden_dims,
        ).to(device)
    elif args.generator_type == "conv":
        if args.encoding == "coarse_residual":
            raise ValueError("--generator_type conv does not support coarse_residual")
        g = ConvDecoderGenerator(
            latent_dim=args.latent_dim,
            output_height=code_height,
            output_width=code_width,
            channels=args.generator_channels,
        ).to(device)
    elif args.generator_type == "set_conv":
        if args.encoding == "coarse_residual":
            raise ValueError(
                "--generator_type set_conv does not support coarse_residual"
            )
        g = SetTransformerConvGenerator(
            latent_dim=args.latent_dim,
            output_height=code_height,
            output_width=code_width,
            channels=args.generator_channels,
            model_dim=args.set_generator_dim,
            depth=args.set_generator_depth,
            heads=args.set_generator_heads,
            mlp_ratio=args.set_generator_mlp_ratio,
            dropout=args.set_generator_dropout,
        ).to(device)
    elif args.generator_type == "set_direct":
        g = SetTransformerDirectGenerator(
            latent_dim=args.latent_dim,
            output_dim=f_dim,
            model_dim=args.set_generator_dim,
            depth=args.set_generator_depth,
            heads=args.set_generator_heads,
            mlp_ratio=args.set_generator_mlp_ratio,
            dropout=args.set_generator_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")
    if args.generator_output_norm != "none":
        g = NormalizedGenerator(g, mode=args.generator_output_norm).to(device)

    discriminator_output_dim = (
        2 if args.optimizer_type == "quantile_ranked_value_default" else 1
    )
    if args.discriminator_type == "mlp":
        if args.optimizer_type == "quantile_ranked_value_default":
            d = RankValueMLP(
                input_dim=d_input_dim,
                hidden_dims=args.discriminator_hidden_dims,
                use_spectral_norm=args.discriminator_spectral_norm,
            ).to(device)
        else:
            d = MLP(
                input_dim=d_input_dim,
                output_dim=discriminator_output_dim,
                hidden_dims=args.discriminator_hidden_dims,
                use_spectral_norm=args.discriminator_spectral_norm,
            ).to(device)
    elif args.discriminator_type == "conv":
        if args.train_on_decoded:
            d_height = args.grid_height
            d_width = args.grid_width
        else:
            if f_dim != code_height * code_width:
                raise ValueError(
                    "--discriminator_type conv requires a grid-shaped discriminator input"
                )
            d_height = code_height
            d_width = code_width
        d = ConvDiscriminator(
            input_height=d_height,
            input_width=d_width,
            channels=args.discriminator_channels,
            use_spectral_norm=args.discriminator_spectral_norm,
            activation=args.discriminator_activation,
            output_dim=discriminator_output_dim,
        ).to(device)
    elif args.discriminator_type == "set_transformer":
        d = SetTransformerDiscriminator(
            input_dim=d_input_dim,
            model_dim=args.set_discriminator_dim,
            depth=args.set_discriminator_depth,
            heads=args.set_discriminator_heads,
            mlp_ratio=args.set_discriminator_mlp_ratio,
            dropout=args.set_discriminator_dropout,
            output_dim=discriminator_output_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown discriminator_type: {args.discriminator_type}")

    if args.removal_ladder_volumes:
        stage_names = []
        for volume in args.removal_ladder_volumes:
            if args.removal_ladder_connectivity_max is not None:
                stage_names.append(f"removal_conn_v{volume:g}")
            stage_names.append(f"removal_comp_v{volume:g}")
        value_levels = Levels(
            [*stage_names, "volume_violation", "roughness_violation", "compliance"]
        )
    elif levels_ladder_rungs:
        objective_names = [rung.name for rung in levels_ladder_rungs]
        final_open = args.levels_ladder_final_open
        if final_open == "":
            final_open = None
        elif final_open not in objective_names:
            final_open = None
        value_levels = Levels.ladder(
            levels_ladder_rungs,
            interleave=True,
            final_open=final_open,
        )
    else:
        level_names = get_level_names(
            args.volume_ladder,
            args.compliance_ladder,
            args.roughness_ladder,
            args.connectivity_ladder,
            args.diversity_ladder,
            args.connectivity_max,
            ladder_sequence,
        )
        value_levels = Levels(level_names)
    buffer_size = args.buffer_multiplier * args.batch_size
    if args.niche_buffer_count > 1:
        if proposal_design_proxy is None:
            raise RuntimeError("niche buffer requires a decoded-design proxy")
        buffer_impl = NicheEliteBuffer(
            buffer_size=buffer_size,
            value_levels=value_levels,
            niche_count=args.niche_buffer_count,
            design_proxy=proposal_design_proxy,
            min_hamming=args.niche_buffer_min_hamming,
            view_mode=args.niche_buffer_view_mode,
            cross_niche_min_hamming=args.niche_buffer_cross_min_hamming,
            cross_niche_reference_top_k=args.niche_buffer_cross_reference_top_k,
        )
    else:
        base_buffer = Buffer(
            buffer_size=buffer_size,
            value_levels=value_levels,
        )
        if args.buffer_diversity_min_hamming > 0:
            buffer_impl = DiverseEliteBuffer(
                base_buffer,
                min_hamming=args.buffer_diversity_min_hamming,
                topk_frac=args.buffer_diversity_topk_frac,
            )
        else:
            buffer_impl = base_buffer
    buffer = components.BufferComp(
        B=buffer_impl,
    )
    objective = BufferChamferDiversityObjective(evaluator, buffer.B)
    fn = components.Fn(
        f=objective,
        input_dim=f_dim,
        device=device,
        dtype=torch.float32,
    )

    curiosity_scheduler = None
    if args.curiosity > 0 and args.curiosity_schedule == "warmup_cosine":
        curiosity_scheduler = WarmupCosine(
            total_steps=args.n_iter,
            warmup_frac=args.curiosity_warmup_frac,
            base=1.0,
            min_val=args.curiosity_min,
        )
    elif args.curiosity > 0 and args.curiosity_schedule == "warmup_cosine_annealing":
        curiosity_scheduler = WarmupCosineAnnealing(
            total_steps=args.n_iter,
            cycles=args.curiosity_cycles,
            warmup_frac=args.curiosity_warmup_frac,
            base=1.0,
            min_val=args.curiosity_min,
            decay=parse_schedule_decay(args.curiosity_decay),
        )
    elif args.curiosity > 0 and args.curiosity_schedule == "cosine_ramp":
        curiosity_scheduler = CosineRamp(
            total_steps=args.n_iter,
            base=1.0,
            min_val=args.curiosity_min,
        )
    elif args.curiosity_schedule != "none":
        raise ValueError(f"Unknown curiosity_schedule: {args.curiosity_schedule}")

    curiosity_loss = None
    if args.curiosity > 0:
        curiosity_use_buffer = args.curiosity_reference == "buffer"
        if args.curiosity_space == "raw":
            curiosity_loss = WangIsolaUniformity(
                WangIsolaUniformityConfig(
                    use_buffer=curiosity_use_buffer,
                    weight=args.curiosity,
                ),
                buffer=buffer.B,
                scheduler=curiosity_scheduler,
            )
        elif args.curiosity_space == "topology":
            curiosity_loss = TopologySpaceUniformity(
                evaluator=evaluator,
                buffer=buffer.B,
                weight=args.curiosity,
                use_buffer=curiosity_use_buffer,
                scheduler=curiosity_scheduler,
            )
        elif args.curiosity_space == "plummer":
            curiosity_loss = PlummerEmbeddingRepulsion(
                buffer=buffer.B,
                weight=args.curiosity,
                power=args.plummer_power,
                eps=args.plummer_eps,
                normalize=args.plummer_normalize,
                terms=args.plummer_terms,
                use_buffer=curiosity_use_buffer,
                scheduler=curiosity_scheduler,
            )
        else:
            raise ValueError(f"Unknown curiosity_space: {args.curiosity_space}")

    if args.niche_output_separation_weight > 0:
        if proposal_design_proxy is None:
            raise RuntimeError(
                "niche output separation requires a decoded-design proxy"
            )
        existing_losses = []
        if curiosity_loss is not None:
            existing_losses.append(curiosity_loss)
        existing_losses.append(
            NicheOutputSeparationLoss(
                buffer=buffer.B,
                design_proxy=proposal_design_proxy,
                weight=args.niche_output_separation_weight,
                margin=args.niche_output_separation_margin,
            )
        )
        curiosity_loss = CombinedCuriosityLoss(existing_losses)

    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
        latent_dim=args.latent_dim,
        optimizerG=make_torch_optimizer(
            args.g_torch_optimizer,
            g.parameters(),
            lr=args.g_lr,
            momentum=args.g_momentum,
        ),
        optimizerD=make_torch_optimizer(
            args.d_torch_optimizer,
            d.parameters(),
            lr=args.d_lr,
            momentum=args.d_momentum,
        ),
        latent_sampler=LatentSamplerLambda(
            lambda b, d, distribution, uniform_low, uniform_high: sample_latents(
                b,
                d,
                distribution=distribution,
                uniform_low=uniform_low,
                uniform_high=uniform_high,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
            b=args.batch_size,
            d=args.latent_dim,
            distribution=args.latent_distribution,
            uniform_low=args.latent_uniform_low,
            uniform_high=args.latent_uniform_high,
        ),
        device=device,
        dtype=torch.float32,
    )
    warmup_batch_size = args.g_uniformity_warmup_batch_size or args.batch_size
    warmup_weight = (
        args.g_uniformity_warmup_weight
        if args.g_uniformity_warmup_weight is not None
        else (args.curiosity if args.curiosity > 0 else 1.0)
    )
    run_generator_uniformity_warmup(
        gan,
        steps=args.g_uniformity_warmup_steps,
        batch_size=warmup_batch_size,
        weight=warmup_weight,
        t=args.g_uniformity_warmup_t,
    )
    fixed_latent_bank_size = 0
    fixed_latent_uniformity_active = False
    fixed_latent_niche_count = 0
    fixed_latent_niche_labels_np = np.empty(0, dtype=np.int32)
    fixed_latent_niche_stats: dict[str, float] = {}
    if args.fixed_latent_bank:
        if not isinstance(gan.latent_dim, int):
            raise ValueError("--fixed_latent_bank requires an integer latent_dim")
        fixed_latent_bank_size = (
            args.fixed_latent_bank_size
            if args.fixed_latent_bank_size is not None
            else buffer.B.buffer_size
        )
        latent_cluster_labels: torch.Tensor | None = None
        if args.fixed_latent_selection == "output_diverse":
            latent_bank = select_output_diverse_latent_bank(
                gan.G,
                latent_dim=gan.latent_dim,
                bank_size=fixed_latent_bank_size,
                candidate_multiplier=args.fixed_latent_candidate_multiplier,
                chunk_size=args.fixed_latent_chunk_size,
                distribution=args.latent_distribution,
                uniform_low=args.latent_uniform_low,
                uniform_high=args.latent_uniform_high,
                device=device,
                dtype=torch.float32,
            )
        elif args.fixed_latent_selection == "clustered_niches":
            fixed_latent_niche_count = args.fixed_latent_niche_count or max(
                args.niche_buffer_count, 2
            )
            (
                latent_bank,
                latent_cluster_labels,
                fixed_latent_niche_stats,
            ) = make_clustered_niche_latent_bank(
                latent_dim=gan.latent_dim,
                bank_size=fixed_latent_bank_size,
                niche_count=fixed_latent_niche_count,
                center_scale=args.fixed_latent_niche_center_scale,
                within_std=args.fixed_latent_niche_within_std,
                normalize_radius=not args.fixed_latent_niche_no_normalize_radius,
                device=device,
                dtype=torch.float32,
            )
            fixed_latent_niche_labels_np = (
                latent_cluster_labels.detach().cpu().numpy().astype(np.int32)
            )
        elif args.fixed_latent_selection == "random":
            latent_bank = sample_latents(
                fixed_latent_bank_size,
                gan.latent_dim,
                distribution=args.latent_distribution,
                uniform_low=args.latent_uniform_low,
                uniform_high=args.latent_uniform_high,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown fixed_latent_selection: {args.fixed_latent_selection}"
            )
        gan.latent_sampler = FixedLatentBankSampler(
            latent_bank,
            batch_size=args.batch_size,
            mode=args.fixed_latent_sample_mode,
            cluster_labels=latent_cluster_labels,
            noise_std=args.fixed_latent_noise_std,
            normalize_noise_scale=not args.fixed_latent_noise_no_normalize,
        )
        if args.fixed_latent_uniformity_weight > 0:
            existing_losses = []
            if gan.curiosity_loss is not None:
                existing_losses.append(gan.curiosity_loss)
            existing_losses.append(
                FixedLatentBankUniformity(
                    gan.G,
                    latent_bank,
                    weight=args.fixed_latent_uniformity_weight,
                    batch_size=args.fixed_latent_uniformity_batch_size,
                    t=args.fixed_latent_uniformity_t,
                    sample_mode=args.fixed_latent_uniformity_sample_mode,
                )
            )
            gan.curiosity_loss = CombinedCuriosityLoss(existing_losses)
            fixed_latent_uniformity_active = True
        fixed_latent_niche_stats_text = " ".join(
            f"{key}={value:.4g}" for key, value in fixed_latent_niche_stats.items()
        )
        logger.info(
            "Using fixed latent bank: "
            f"size={fixed_latent_bank_size} selection={args.fixed_latent_selection} "
            f"candidate_multiplier={args.fixed_latent_candidate_multiplier} "
            f"sample_mode={args.fixed_latent_sample_mode} "
            f"niche_count={fixed_latent_niche_count} "
            f"niche_center_scale={args.fixed_latent_niche_center_scale} "
            f"niche_within_std={args.fixed_latent_niche_within_std} "
            f"niche_normalize_radius={not args.fixed_latent_niche_no_normalize_radius} "
            f"{fixed_latent_niche_stats_text} "
            f"noise_std={args.fixed_latent_noise_std} "
            f"noise_normalize={not args.fixed_latent_noise_no_normalize} "
            f"uniformity_weight={args.fixed_latent_uniformity_weight} "
            f"uniformity_batch_size={args.fixed_latent_uniformity_batch_size} "
            f"uniformity_sample_mode={args.fixed_latent_uniformity_sample_mode}"
        )
    elif args.fixed_latent_uniformity_weight > 0:
        raise ValueError(
            "--fixed_latent_uniformity_weight requires --fixed_latent_bank"
        )

    original_generator = gan.G
    initial_replay_generator: InitialBufferReplayGenerator | None = None
    initial_blob_labels_np = np.empty(0, dtype=np.int32)
    initial_blob_stats: dict[str, float] = {}
    if args.initial_buffer_mode == "random_blobs":
        initial_fill_count = (
            math.ceil(buffer.B.buffer_size / args.batch_size) * args.batch_size
        )
        cluster_niche_count = (
            args.niche_buffer_count if args.initial_blob_cluster_niches else 1
        )
        seed_codes_np, seed_stats, initial_blob_labels_np = (
            make_sorted_material_blob_seed_codes(
                initial_fill_count,
                nely=args.grid_height,
                nelx=args.grid_width,
                target_mean=args.volume_max,
                seed=args.seed + 1009,
                blob_count_min=args.initial_blob_count_min,
                blob_count_max=args.initial_blob_count_max,
                radius_min=args.initial_blob_radius_min,
                radius_max=args.initial_blob_radius_max,
                score_margin=args.initial_blob_score_margin,
                score_noise=args.initial_blob_score_noise,
                candidate_multiplier=args.initial_blob_candidate_multiplier,
                min_hamming=args.initial_blob_min_hamming,
                cluster_niche_count=cluster_niche_count,
            )
        )
        initial_blob_stats = seed_stats
        if args.initial_blob_cluster_niches:
            if not hasattr(buffer.B, "queue_initial_niche_labels"):
                raise RuntimeError(
                    "--initial_blob_cluster_niches requires NicheEliteBuffer"
                )
            buffer.B.queue_initial_niche_labels(initial_blob_labels_np)
        seed_codes = torch.from_numpy(seed_codes_np).to(
            device=device, dtype=torch.float32
        )
        initial_replay_generator = InitialBufferReplayGenerator(
            seed_codes,
            fallback=original_generator,
        ).to(device)
        gan.G = initial_replay_generator
        cluster_log = ""
        if args.initial_blob_cluster_niches:
            cluster_log = (
                f" cluster_niches={cluster_niche_count} "
                f"cluster_internal_mean={seed_stats.get('cluster_internal_mean_hamming', float('nan')):.4f} "
                f"cluster_external_mean={seed_stats.get('cluster_external_mean_hamming', float('nan')):.4f} "
                f"cluster_margin={seed_stats.get('cluster_separation_margin', float('nan')):.4f}"
            )
        logger.info(
            "Using random blob initial buffer: "
            f"count={initial_fill_count} blobs=[{args.initial_blob_count_min},"
            f"{args.initial_blob_count_max}] radius=[{args.initial_blob_radius_min:g},"
            f"{args.initial_blob_radius_max:g}] score_margin={args.initial_blob_score_margin:g} "
            f"score_noise={args.initial_blob_score_noise:g} "
            f"candidate_multiplier={args.initial_blob_candidate_multiplier} "
            f"target_min_hamming={args.initial_blob_min_hamming:g} "
            f"actual_min_hamming={seed_stats['min_pairwise_hamming']:.4f} "
            f"actual_mean_hamming={seed_stats['mean_pairwise_hamming']:.4f}"
            f"{cluster_log}"
        )

    opt_components = components.OptComponents(
        fn=fn,
        gan=gan,
        batch_size=args.batch_size,
        buffer=buffer,
        discriminator_steps=args.discriminator_steps,
        elite_sampling=args.elite_sampling,
        elite_pool_size=args.elite_pool_size
        if args.elite_pool_size is not None
        else args.buffer_multiplier * args.batch_size,
        weight_clip=args.weight_clip
        if args.optimizer_type in {"wgan", "ranked_wgan"}
        else None,
        gradient_penalty_weight=args.gradient_penalty_weight,
    )
    try:
        if args.train_on_decoded:
            if args.optimizer_type == "plackett_luce":
                raise ValueError(
                    "--optimizer_type plackett_luce does not support --train_on_decoded yet"
                )
            optimizer = make_decoded_density_optimizer(
                args.optimizer_type,
                opt_components,
                evaluator=evaluator,
                objective=objective,
            )
        else:
            optimizer = make_optimizer(
                args.optimizer_type,
                opt_components,
                archive_size=args.ranker_archive_size,
                ranker_list_size=args.ranker_list_size,
                ranker_steps=args.ranker_steps,
                ranker_generator_elite_margin=args.ranker_generator_elite_margin,
                ranker_weight=args.ranker_weight,
                ranker_target_curve=args.ranker_target_curve,
                ranker_tau=args.ranker_tau,
                ranker_sample_pool_size=args.ranker_sample_pool_size,
                ranker_sample_mode=args.ranker_sample_mode,
                ranker_target_scope=args.ranker_target_scope,
                ranker_niche_local_targets=args.ranker_niche_local_targets,
                ranker_list_repeats=args.ranker_list_repeats,
                ranker_fake_weight=args.ranker_fake_weight,
                ranker_fake_repeats=args.ranker_fake_repeats,
                d_score_center_weight=args.d_score_center_weight,
                d_score_scale_weight=args.d_score_scale_weight,
                d_score_target_std=args.d_score_target_std,
                utility_target_scale=args.utility_target_scale,
                utility_loss=args.utility_loss,
                utility_weight=args.utility_weight,
                generator_utility_weight=args.generator_utility_weight,
                utility_clip=args.utility_clip,
                proposal_pool_size=args.proposal_pool_size,
                proposal_top_k=args.proposal_top_k,
                proposal_diversity_min_hamming=args.proposal_diversity_min_hamming,
                proposal_diversity_topk_frac=args.proposal_diversity_topk_frac,
                proposal_buffer_novelty_min_hamming=args.proposal_buffer_novelty_min_hamming,
                proposal_buffer_novelty_reference_size=args.proposal_buffer_novelty_reference_size,
                proposal_buffer_reject_exact_design_duplicates=args.proposal_buffer_reject_exact_design_duplicates,
                design_proxy=proposal_design_proxy,
                proposal_evolution_fraction=args.proposal_evolution_fraction,
                proposal_evolution_parent_source=args.proposal_evolution_parent_source,
                proposal_evolution_crossover=args.proposal_evolution_crossover,
                proposal_evolution_mutation_rate=args.proposal_evolution_mutation_rate,
                proposal_evolution_mutation_scale=args.proposal_evolution_mutation_scale,
                proposal_evolution_grid_height=code_height
                if f_dim == code_height * code_width
                else None,
                proposal_evolution_grid_width=code_width
                if f_dim == code_height * code_width
                else None,
                proposal_gradient_steps=args.proposal_gradient_steps,
                proposal_gradient_step_size=args.proposal_gradient_step_size,
                proposal_gradient_mode=args.proposal_gradient_mode,
                proposal_gradient_normalize=not args.proposal_gradient_no_normalize,
                proposal_gradient_noise=args.proposal_gradient_noise,
                proposal_gradient_keep_original=args.proposal_gradient_keep_original,
                ga_offspring_fraction=args.ga_offspring_fraction,
                ga_pool_size=args.ga_pool_size,
                ga_parent_pool_size=args.ga_parent_pool_size,
                ga_mutation_rate=args.ga_mutation_rate,
                ga_mutation_scale=args.ga_mutation_scale,
                generator_elite_context_size=args.set_generator_elite_context_size,
                generator_elite_context_pool_size=args.set_generator_elite_context_pool_size,
            )
    finally:
        if initial_replay_generator is not None:
            gan.G = original_generator

    density_filter_schedule_active = (
        args.density_filter_warmup_iters > 0
        and args.density_filter_warmup_iters < args.n_iter
        and args.density_filter_final_radius != args.density_filter_radius
    )
    density_filter_switched = False
    density_filter_extra_eval_count = 0
    if args.density_filter_warmup_iters > 0:
        logger.info(
            "Density filter schedule: "
            f"warmup_radius={args.density_filter_radius} "
            f"warmup_iters={args.density_filter_warmup_iters} "
            f"final_radius={args.density_filter_final_radius} "
            f"will_switch={density_filter_schedule_active}"
        )

    logger.info(
        f"FEMCantilever: preset={args.preset} grid={args.grid_width}x{args.grid_height} domain={args.domain_width:g}x{args.domain_height:g} code_grid={code_width}x{code_height} backend={args.backend} encoding={args.encoding} optimizer={args.optimizer_type} n_iter={args.n_iter} "
        f"latent_distribution={args.latent_distribution} latent_uniform=[{args.latent_uniform_low:g},{args.latent_uniform_high:g}] "
        f"initial_buffer_mode={args.initial_buffer_mode} initial_blob_count=[{args.initial_blob_count_min},{args.initial_blob_count_max}] initial_blob_radius=[{args.initial_blob_radius_min:g},{args.initial_blob_radius_max:g}] initial_blob_score_margin={args.initial_blob_score_margin:g} initial_blob_score_noise={args.initial_blob_score_noise:g} initial_blob_candidate_multiplier={args.initial_blob_candidate_multiplier} initial_blob_min_hamming={args.initial_blob_min_hamming:g} "
        f"G={args.generator_type} G_norm={args.generator_output_norm} D={args.discriminator_type} "
        f"D_spectral_norm={args.discriminator_spectral_norm} D_activation={args.discriminator_activation} "
        f"fixed_latent_bank={args.fixed_latent_bank} fixed_latent_bank_size={fixed_latent_bank_size} fixed_latent_selection={args.fixed_latent_selection} fixed_latent_sample_mode={args.fixed_latent_sample_mode} fixed_latent_niche_count={fixed_latent_niche_count} fixed_latent_niche_center_scale={args.fixed_latent_niche_center_scale} fixed_latent_niche_within_std={args.fixed_latent_niche_within_std} fixed_latent_niche_normalize_radius={not args.fixed_latent_niche_no_normalize_radius} fixed_latent_noise_std={args.fixed_latent_noise_std} fixed_latent_noise_normalize={not args.fixed_latent_noise_no_normalize} fixed_latent_uniformity_active={fixed_latent_uniformity_active} fixed_latent_uniformity_weight={args.fixed_latent_uniformity_weight} fixed_latent_uniformity_batch_size={args.fixed_latent_uniformity_batch_size} "
        f"setG_dim={args.set_generator_dim} setG_depth={args.set_generator_depth} setG_heads={args.set_generator_heads} setG_elite_context={args.set_generator_elite_context_size} setG_elite_pool={args.set_generator_elite_context_pool_size} "
        f"setD_dim={args.set_discriminator_dim} setD_depth={args.set_discriminator_depth} setD_heads={args.set_discriminator_heads} "
        f"curiosity={args.curiosity} curiosity_space={args.curiosity_space} curiosity_reference={args.curiosity_reference} curiosity_schedule={args.curiosity_schedule} curiosity_cycles={args.curiosity_cycles} curiosity_decay={args.curiosity_decay} plummer_power={args.plummer_power} plummer_eps={args.plummer_eps} plummer_normalize={args.plummer_normalize} plummer_terms={args.plummer_terms} g_uniformity_warmup_steps={args.g_uniformity_warmup_steps} g_uniformity_warmup_batch_size={warmup_batch_size} g_uniformity_warmup_weight={warmup_weight} g_opt={args.g_torch_optimizer} d_opt={args.d_torch_optimizer} g_lr={args.g_lr} d_lr={args.d_lr} buffer_diversity_min_hamming={args.buffer_diversity_min_hamming} buffer_diversity_topk_frac={args.buffer_diversity_topk_frac} niche_buffer_count={args.niche_buffer_count} niche_buffer_min_hamming={args.niche_buffer_min_hamming} niche_buffer_view_mode={args.niche_buffer_view_mode} niche_buffer_cross_min_hamming={args.niche_buffer_cross_min_hamming} niche_buffer_cross_reference_top_k={args.niche_buffer_cross_reference_top_k} niche_output_separation_weight={args.niche_output_separation_weight} niche_output_separation_margin={args.niche_output_separation_margin} elite_sampling={args.elite_sampling} elite_pool_size={args.elite_pool_size} ranker_list_size={args.ranker_list_size} ranker_steps={args.ranker_steps} ranker_weight={args.ranker_weight} ranker_target_curve={args.ranker_target_curve} ranker_tau={args.ranker_tau} ranker_target_scope={args.ranker_target_scope} ranker_niche_local_targets={args.ranker_niche_local_targets} ranker_list_repeats={args.ranker_list_repeats} ranker_fake_weight={args.ranker_fake_weight} ranker_fake_repeats={args.ranker_fake_repeats} ranker_sample_pool_size={args.ranker_sample_pool_size} ranker_sample_mode={args.ranker_sample_mode} d_score_center_weight={args.d_score_center_weight} d_score_scale_weight={args.d_score_scale_weight} d_score_target_std={args.d_score_target_std} utility_target_scale={args.utility_target_scale} utility_loss={args.utility_loss} utility_weight={args.utility_weight} generator_utility_weight={args.generator_utility_weight} utility_clip={args.utility_clip} proposal_pool_size={args.proposal_pool_size} proposal_top_k={args.proposal_top_k} proposal_diversity_min_hamming={args.proposal_diversity_min_hamming} proposal_buffer_novelty_min_hamming={args.proposal_buffer_novelty_min_hamming} proposal_buffer_novelty_reference_size={args.proposal_buffer_novelty_reference_size} proposal_buffer_reject_exact_design_duplicates={args.proposal_buffer_reject_exact_design_duplicates} proposal_buffer_novelty_threshold={args.proposal_buffer_novelty_threshold} proposal_evolution_fraction={args.proposal_evolution_fraction} proposal_evolution_parent_source={args.proposal_evolution_parent_source} proposal_evolution_crossover={args.proposal_evolution_crossover} proposal_evolution_mutation_rate={args.proposal_evolution_mutation_rate} proposal_evolution_mutation_scale={args.proposal_evolution_mutation_scale} proposal_gradient_steps={args.proposal_gradient_steps} proposal_gradient_step_size={args.proposal_gradient_step_size} proposal_gradient_mode={args.proposal_gradient_mode} proposal_gradient_normalize={not args.proposal_gradient_no_normalize} proposal_gradient_noise={args.proposal_gradient_noise} proposal_gradient_keep_original={args.proposal_gradient_keep_original} ga_offspring_fraction={args.ga_offspring_fraction} ga_pool_size={args.ga_pool_size} ga_parent_pool_size={args.ga_parent_pool_size} ga_mutation_rate={args.ga_mutation_rate} ga_mutation_scale={args.ga_mutation_scale} load_case={args.load_case} robust_load_cases={args.robust_load_cases} robust_load_aggregate={args.robust_load_aggregate} robust_load_cvar_frac={args.robust_load_cvar_frac} removal_ladder_volumes={args.removal_ladder_volumes} removal_ladder_compliances={args.removal_ladder_compliances} removal_ladder_connectivity_max={args.removal_ladder_connectivity_max} load_scale={args.load_scale} fem_workers={args.fem_workers} compliance_solver={args.compliance_solver} matrix_free_cg_max_iter={args.matrix_free_cg_max_iter} matrix_free_cg_tol={args.matrix_free_cg_tol} matrix_free_cg_device={args.matrix_free_cg_device} matrix_free_cg_dtype={args.matrix_free_cg_dtype} filter_radius={args.density_filter_radius} filter_warmup_iters={args.density_filter_warmup_iters} filter_final_radius={args.density_filter_final_radius} residual_scale={args.residual_scale} "
        f"projection_beta={args.projection_beta} hard_binarize={args.hard_binarize} binhead_connect_support={args.binhead_connect_support} train_on_decoded={args.train_on_decoded}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"curiosity_{args.curiosity:g}_seed_{args.seed}"
    live_progress_path = args.output_dir / f"live_progress_{suffix}.jsonl"
    live_latest_path = args.output_dir / f"live_progress_latest_{suffix}.json"
    live_progress_enabled = not args.no_live_progress
    live_start_time = time.perf_counter()
    if live_progress_enabled:
        live_progress_path.write_text("", encoding="utf-8")

    init_eval_count = (
        math.ceil(buffer.B.buffer_size / args.batch_size) * args.batch_size
    )
    history = [
        record_buffer_history(
            buffer.B,
            iteration=0,
            eval_count=init_eval_count,
        )
    ]
    niche_history: list[np.ndarray] = [
        record_niche_history(
            buffer.B,
            iteration=0,
            eval_count=init_eval_count,
        )
    ]
    design_history: list[np.ndarray] = []
    design_history_values: list[np.ndarray] = []
    design_history_raw_code: list[np.ndarray] = []
    design_history_iterations: list[float] = []
    design_history_eval_counts: list[float] = []

    def write_live_progress(row: dict[str, np.ndarray | float]) -> None:
        if not live_progress_enabled:
            return
        payload = live_progress_payload(
            row,
            start_time=live_start_time,
            n_iter=args.n_iter,
        )
        with live_progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        tmp_latest_path = live_latest_path.with_suffix(live_latest_path.suffix + ".tmp")
        tmp_latest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_latest_path.replace(live_latest_path)
        eta_min = payload["eta_sec"] / 60.0
        logger.info(
            "Progress "
            f"iter={payload['iteration']}/{payload['n_iter']} "
            f"evals={payload['eval_count']} "
            f"best={payload['best_last']:.6g} "
            f"best_feasible={payload['best_feasible_last']} "
            f"mean={payload['mean_last']:.6g} "
            f"feasible={payload['feasible_rate']} "
            f"eta_min={eta_min:.1f} "
            f"live={live_latest_path}"
        )

    def record_design_history_checkpoint(iteration: int, eval_count: int) -> None:
        if args.design_history_top_k <= 0:
            return
        history_top_k = min(args.design_history_top_k, len(buffer.B))
        if history_top_k <= 0:
            return
        tensors = buffer.B.get_top_k(history_top_k).detach().cpu()
        values = np.asarray(
            buffer.B.get_sorted_values()[:history_top_k],
            dtype=np.float32,
        )
        if args.train_on_decoded:
            designs_np = tensors.reshape(
                history_top_k,
                args.grid_height,
                args.grid_width,
            ).numpy()
        else:
            designs_np = evaluator.decode_designs_numpy(tensors.numpy())
        design_history.append(designs_np.astype(np.float32, copy=False))
        design_history_values.append(values)
        design_history_raw_code.append(tensors.numpy().astype(np.float32, copy=False))
        design_history_iterations.append(float(iteration))
        design_history_eval_counts.append(float(eval_count))

    def switch_density_filter_if_due(iteration: int) -> None:
        nonlocal density_filter_switched, density_filter_extra_eval_count
        if (
            not density_filter_schedule_active
            or density_filter_switched
            or iteration <= args.density_filter_warmup_iters
        ):
            return

        previous_radius = evaluator.config.density_filter_radius
        evaluator.set_density_filter_radius(args.density_filter_final_radius)
        reeval_count = reevaluate_buffer_entries(
            buffer.B,
            fn.f,
            device=fn.device,
            dtype=fn.dtype,
            batch_size=args.batch_size,
        )
        density_filter_extra_eval_count += reeval_count
        density_filter_switched = True
        switch_iteration = iteration - 1
        switch_eval_count = (
            init_eval_count
            + switch_iteration * args.batch_size
            + density_filter_extra_eval_count
        )
        logger.info(
            "Switched density filter radius "
            f"from {previous_radius} to {args.density_filter_final_radius} "
            f"after iteration={switch_iteration}; "
            f"re_evaluated_buffer_entries={reeval_count} "
            f"eval_count={switch_eval_count}"
        )
        history_row = record_buffer_history(
            buffer.B,
            iteration=switch_iteration,
            eval_count=switch_eval_count,
        )
        history.append(history_row)
        niche_history.append(
            record_niche_history(
                buffer.B,
                iteration=switch_iteration,
                eval_count=switch_eval_count,
            )
        )
        write_live_progress(history_row)
        record_design_history_checkpoint(
            iteration=switch_iteration,
            eval_count=switch_eval_count,
        )

    write_live_progress(history[-1])
    record_design_history_checkpoint(iteration=0, eval_count=init_eval_count)
    progress = Progress(
        TextColumn("Iteration {task.completed}"),
        BarColumn(),
        TextColumn("Best: {task.fields[best]:.4f}"),
        TextColumn("Mean: {task.fields[mean]:.4f}"),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task(
            "Optimizing",
            total=args.n_iter,
            best=buffer.B.get_value(0, level=-1),
            mean=buffer.B.get_mean_buffer_value(level=-1),
        )
        for iteration in range(1, args.n_iter + 1):
            switch_density_filter_if_due(iteration)
            optimizer.step()
            eval_count = (
                init_eval_count
                + iteration * args.batch_size
                + density_filter_extra_eval_count
            )
            if iteration % args.history_interval == 0 or iteration == args.n_iter:
                history_row = record_buffer_history(
                    buffer.B,
                    iteration=iteration,
                    eval_count=eval_count,
                )
                history.append(history_row)
                niche_history.append(
                    record_niche_history(
                        buffer.B,
                        iteration=iteration,
                        eval_count=eval_count,
                    )
                )
                write_live_progress(history_row)
                record_design_history_checkpoint(
                    iteration=iteration,
                    eval_count=eval_count,
                )
            progress.update(
                task,
                advance=1,
                best=buffer.B.get_value(0, level=-1),
                mean=buffer.B.get_mean_buffer_value(level=-1),
            )

    top_k = min(9, len(buffer.B))
    top_archive_tensors = buffer.B.get_top_k(top_k)
    if args.train_on_decoded:
        top_designs = top_archive_tensors.reshape(
            -1, args.grid_height, args.grid_width
        ).to(
            device=device,
            dtype=torch.float32,
        )
        raw_design_code = np.empty((0, f_dim), dtype=np.float32)
    else:
        top_designs = torch.from_numpy(
            evaluator.decode_designs_numpy(top_archive_tensors.detach().cpu().numpy())
        ).to(device=device, dtype=torch.float32)
        raw_design_code = top_archive_tensors.cpu().numpy()
    transformed_archive_values = np.asarray(
        buffer.B.get_sorted_values(),
        dtype=np.float32,
    )
    top_values = transformed_archive_values[:top_k]
    summary_values = transformed_archive_values
    volume_violation = np.maximum(
        top_designs.cpu().numpy().mean(axis=(1, 2)) - args.volume_max, 0.0
    ).astype(np.float32)
    top_designs_np = top_designs.cpu().numpy()
    dx = np.abs(top_designs_np[:, :, 1:] - top_designs_np[:, :, :-1]).mean(axis=(1, 2))
    dy = np.abs(top_designs_np[:, 1:, :] - top_designs_np[:, :-1, :]).mean(axis=(1, 2))
    roughness_violation = np.maximum(0.5 * (dx + dy) - args.roughness_max, 0.0).astype(
        np.float32
    )
    if levels_ladder_rungs:
        raw_top_values = []
        for sample in top_designs_np:
            volume, roughness, _connectivity, _diversity, compliance = (
                evaluator.density_objectives(sample)
            )
            raw_top_values.append(
                [
                    max(volume - args.volume_max, 0.0),
                    max(roughness - args.roughness_max, 0.0),
                    compliance,
                ]
            )
        top_values = np.asarray(raw_top_values, dtype=np.float32)
        summary_values = top_values
    metrics = compliance_summary(summary_values)
    actual_compliance = top_values[:, -1].copy()
    relative_compliance = actual_compliance / evaluator.solid_compliance
    per_niche_top_k = 6
    per_niche_top_designs = np.empty(
        (0, 0, args.grid_height, args.grid_width),
        dtype=np.float32,
    )
    per_niche_top_values = np.empty((0, 0, top_values.shape[1]), dtype=np.float32)
    if hasattr(buffer.B, "get_niche_top_k") and hasattr(
        buffer.B,
        "get_niche_sorted_values",
    ):
        niche_count = int(buffer.B.niche_count)
        value_dim = int(transformed_archive_values.shape[1])
        per_niche_top_designs = np.full(
            (niche_count, per_niche_top_k, args.grid_height, args.grid_width),
            np.nan,
            dtype=np.float32,
        )
        per_niche_top_values = np.full(
            (niche_count, per_niche_top_k, value_dim),
            np.nan,
            dtype=np.float32,
        )
        for niche_idx in range(niche_count):
            niche_tensors = buffer.B.get_niche_top_k(niche_idx, per_niche_top_k)
            niche_k = int(niche_tensors.shape[0])
            if niche_k == 0:
                continue
            if args.train_on_decoded:
                niche_designs = (
                    niche_tensors.reshape(
                        niche_k,
                        args.grid_height,
                        args.grid_width,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                niche_designs = evaluator.decode_designs_numpy(
                    niche_tensors.detach().cpu().numpy()
                )
            niche_values = np.asarray(
                buffer.B.get_niche_sorted_values(niche_idx)[:niche_k],
                dtype=np.float32,
            )
            per_niche_top_designs[niche_idx, :niche_k] = niche_designs
            per_niche_top_values[niche_idx, :niche_k] = niche_values

    history_scalar_array, history_best_values, history_columns = history_arrays(history)
    niche_history_array = (
        np.concatenate([row for row in niche_history if row.size], axis=0)
        if any(row.size for row in niche_history)
        else np.empty((0, len(NICHE_HISTORY_COLUMNS)), dtype=np.float32)
    )
    if design_history:
        design_history_array = np.stack(design_history, axis=0).astype(np.float32)
        design_history_values_array = np.stack(design_history_values, axis=0).astype(
            np.float32
        )
        design_history_raw_code_array = np.stack(
            design_history_raw_code,
            axis=0,
        ).astype(np.float32)
    else:
        design_history_array = np.empty(
            (0, 0, args.grid_height, args.grid_width),
            dtype=np.float32,
        )
        design_history_values_array = np.empty((0, 0, 0), dtype=np.float32)
        design_history_raw_code_array = np.empty((0, 0, f_dim), dtype=np.float32)
    design_history_iterations_array = np.asarray(
        design_history_iterations,
        dtype=np.float32,
    )
    design_history_eval_counts_array = np.asarray(
        design_history_eval_counts,
        dtype=np.float32,
    )
    history_plot_path = args.output_dir / f"buffer_history_{suffix}.png"
    best_design_history_plot_path = (
        args.output_dir / f"best_design_history_{suffix}.png"
    )
    per_niche_top_designs_plot_path = (
        args.output_dir / f"top_designs_by_niche_{suffix}.png"
    )
    if not args.no_history_plot:
        plot_buffer_history(
            history,
            history_plot_path,
            title=(
                f"FEM Cantilever Buffer History "
                f"({args.optimizer_type}, {args.encoding}, seed={args.seed})"
            ),
        )
        save_best_design_history_grid(
            design_history_array,
            design_history_values_array,
            design_history_iterations_array,
            best_design_history_plot_path,
            title=(
                f"Best Design History "
                f"({args.optimizer_type}, {args.encoding}, seed={args.seed})"
            ),
        )
    save_design_grid(
        top_designs,
        actual_compliance,
        relative_compliance,
        volume_violation,
        roughness_violation,
        args.output_dir / f"top_designs_{suffix}.png",
        title=(
            f"FEM Cantilever Top Designs (curiosity={args.curiosity:g}, seed={args.seed})"
        ),
    )
    save_per_niche_design_grid(
        per_niche_top_designs,
        per_niche_top_values,
        per_niche_top_designs_plot_path,
        title=(
            f"FEM Cantilever Top Designs By Niche "
            f"(curiosity={args.curiosity:g}, seed={args.seed})"
        ),
    )
    mean_l2 = pairwise_l2_mean(top_designs)
    mean_hamming = pairwise_hamming_mean(top_designs)

    np.savez_compressed(
        args.output_dir / f"top_designs_{suffix}.npz",
        designs=top_designs.cpu().numpy(),
        per_niche_top_designs=per_niche_top_designs,
        per_niche_top_values=per_niche_top_values,
        per_niche_top_designs_plot=np.asarray([str(per_niche_top_designs_plot_path)]),
        archive_tensors=top_archive_tensors.cpu().numpy(),
        raw_design_code=raw_design_code,
        values=top_values,
        all_archive_values=summary_values,
        transformed_archive_values=transformed_archive_values,
        history=history_scalar_array,
        history_columns=history_columns,
        history_best_values=history_best_values,
        niche_history=niche_history_array,
        niche_history_columns=NICHE_HISTORY_COLUMNS,
        history_plot=np.asarray(
            ["" if args.no_history_plot else str(history_plot_path)]
        ),
        history_interval=np.asarray([args.history_interval], dtype=np.int32),
        live_progress=np.asarray(
            ["" if args.no_live_progress else str(live_progress_path)]
        ),
        live_progress_latest=np.asarray(
            ["" if args.no_live_progress else str(live_latest_path)]
        ),
        design_history=design_history_array,
        design_history_values=design_history_values_array,
        design_history_raw_code=design_history_raw_code_array,
        design_history_iterations=design_history_iterations_array,
        design_history_eval_counts=design_history_eval_counts_array,
        design_history_top_k=np.asarray([args.design_history_top_k], dtype=np.int32),
        best_design_history_plot=np.asarray(
            ["" if args.no_history_plot else str(best_design_history_plot_path)]
        ),
        actual_compliance=actual_compliance.astype(np.float32),
        relative_compliance=relative_compliance.astype(np.float32),
        archive_best_compliance=np.asarray(
            [metrics["archive_best_compliance"]], dtype=np.float32
        ),
        best_feasible_compliance=np.asarray(
            [metrics["best_feasible_compliance"]], dtype=np.float32
        ),
        best_any_compliance=np.asarray(
            [metrics["best_any_compliance"]], dtype=np.float32
        ),
        best_feasible_index=np.asarray(
            [metrics["best_feasible_index"]], dtype=np.int32
        ),
        best_any_index=np.asarray([metrics["best_any_index"]], dtype=np.int32),
        feasible_count=np.asarray([metrics["feasible_count"]], dtype=np.int32),
        feasible_rate=np.asarray([metrics["feasible_rate"]], dtype=np.float32),
        mean_l2=np.asarray([mean_l2], dtype=np.float32),
        mean_hamming=np.asarray([mean_hamming], dtype=np.float32),
        solid_compliance=np.asarray([evaluator.solid_compliance], dtype=np.float32),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        curiosity_space=np.asarray([args.curiosity_space]),
        curiosity_reference=np.asarray([args.curiosity_reference]),
        curiosity_schedule=np.asarray([args.curiosity_schedule]),
        curiosity_cycles=np.asarray([args.curiosity_cycles], dtype=np.int32),
        curiosity_decay=np.asarray(
            ["" if args.curiosity_decay is None else args.curiosity_decay]
        ),
        curiosity_warmup_frac=np.asarray(
            [args.curiosity_warmup_frac], dtype=np.float32
        ),
        curiosity_min=np.asarray([args.curiosity_min], dtype=np.float32),
        plummer_power=np.asarray([args.plummer_power], dtype=np.float32),
        plummer_eps=np.asarray([args.plummer_eps], dtype=np.float32),
        plummer_normalize=np.asarray([args.plummer_normalize]),
        plummer_terms=np.asarray([args.plummer_terms]),
        g_uniformity_warmup_steps=np.asarray(
            [args.g_uniformity_warmup_steps], dtype=np.int32
        ),
        g_uniformity_warmup_batch_size=np.asarray([warmup_batch_size], dtype=np.int32),
        g_uniformity_warmup_weight=np.asarray([warmup_weight], dtype=np.float32),
        g_uniformity_warmup_t=np.asarray(
            [args.g_uniformity_warmup_t], dtype=np.float32
        ),
        seed=np.asarray([args.seed], dtype=np.int32),
        preset=np.asarray([args.preset]),
        grid_width=np.asarray([args.grid_width], dtype=np.int32),
        grid_height=np.asarray([args.grid_height], dtype=np.int32),
        latent_distribution=np.asarray([args.latent_distribution]),
        latent_uniform_low=np.asarray([args.latent_uniform_low], dtype=np.float32),
        latent_uniform_high=np.asarray([args.latent_uniform_high], dtype=np.float32),
        domain_width=np.asarray([args.domain_width], dtype=np.float32),
        domain_height=np.asarray([args.domain_height], dtype=np.float32),
        coarse_grid_width=np.asarray([code_width], dtype=np.int32),
        coarse_grid_height=np.asarray([code_height], dtype=np.int32),
        backend=np.asarray([args.backend]),
        encoding=np.asarray([args.encoding]),
        tiny_decoder_model=np.asarray([args.tiny_decoder_model]),
        tiny_decoder_latent_channels=np.asarray(
            [args.tiny_decoder_latent_channels], dtype=np.int32
        ),
        tiny_decoder_latent_height=np.asarray(
            [args.tiny_decoder_latent_height], dtype=np.int32
        ),
        tiny_decoder_latent_width=np.asarray(
            [args.tiny_decoder_latent_width], dtype=np.int32
        ),
        tiny_decoder_latent_scale=np.asarray(
            [args.tiny_decoder_latent_scale], dtype=np.float32
        ),
        sorted_material_profile=np.asarray([args.sorted_material_profile]),
        sorted_material_steepness=np.asarray(
            [args.sorted_material_steepness], dtype=np.float32
        ),
        initial_buffer_mode=np.asarray([args.initial_buffer_mode]),
        initial_blob_count_min=np.asarray(
            [args.initial_blob_count_min], dtype=np.int32
        ),
        initial_blob_count_max=np.asarray(
            [args.initial_blob_count_max], dtype=np.int32
        ),
        initial_blob_radius_min=np.asarray(
            [args.initial_blob_radius_min], dtype=np.float32
        ),
        initial_blob_radius_max=np.asarray(
            [args.initial_blob_radius_max], dtype=np.float32
        ),
        initial_blob_score_margin=np.asarray(
            [args.initial_blob_score_margin], dtype=np.float32
        ),
        initial_blob_score_noise=np.asarray(
            [args.initial_blob_score_noise], dtype=np.float32
        ),
        initial_blob_candidate_multiplier=np.asarray(
            [args.initial_blob_candidate_multiplier], dtype=np.int32
        ),
        initial_blob_min_hamming=np.asarray(
            [args.initial_blob_min_hamming], dtype=np.float32
        ),
        initial_blob_cluster_niches=np.asarray(
            [args.initial_blob_cluster_niches], dtype=np.int32
        ),
        initial_blob_niche_labels=initial_blob_labels_np.astype(np.int32, copy=False),
        initial_blob_cluster_niche_count=np.asarray(
            [initial_blob_stats.get("cluster_niche_count", 0.0)],
            dtype=np.float32,
        ),
        initial_blob_cluster_internal_mean_hamming=np.asarray(
            [initial_blob_stats.get("cluster_internal_mean_hamming", float("nan"))],
            dtype=np.float32,
        ),
        initial_blob_cluster_external_mean_hamming=np.asarray(
            [initial_blob_stats.get("cluster_external_mean_hamming", float("nan"))],
            dtype=np.float32,
        ),
        initial_blob_cluster_separation_margin=np.asarray(
            [initial_blob_stats.get("cluster_separation_margin", float("nan"))],
            dtype=np.float32,
        ),
        initial_blob_cluster_medoid_min_hamming=np.asarray(
            [initial_blob_stats.get("cluster_medoid_min_hamming", float("nan"))],
            dtype=np.float32,
        ),
        bar_count=np.asarray([args.bar_count], dtype=np.int32),
        bar_width_min=np.asarray([args.bar_width_min], dtype=np.float32),
        bar_width_max=np.asarray([args.bar_width_max], dtype=np.float32),
        bar_edge_softness=np.asarray([args.bar_edge_softness], dtype=np.float32),
        generator_type=np.asarray([args.generator_type]),
        generator_output_norm=np.asarray([args.generator_output_norm]),
        discriminator_type=np.asarray([args.discriminator_type]),
        discriminator_spectral_norm=np.asarray([args.discriminator_spectral_norm]),
        discriminator_activation=np.asarray([args.discriminator_activation]),
        fixed_latent_bank=np.asarray([args.fixed_latent_bank]),
        fixed_latent_bank_size=np.asarray([fixed_latent_bank_size], dtype=np.int32),
        fixed_latent_selection=np.asarray([args.fixed_latent_selection]),
        fixed_latent_candidate_multiplier=np.asarray(
            [args.fixed_latent_candidate_multiplier], dtype=np.int32
        ),
        fixed_latent_sample_mode=np.asarray([args.fixed_latent_sample_mode]),
        fixed_latent_niche_count=np.asarray([fixed_latent_niche_count], dtype=np.int32),
        fixed_latent_niche_center_scale=np.asarray(
            [args.fixed_latent_niche_center_scale], dtype=np.float32
        ),
        fixed_latent_niche_within_std=np.asarray(
            [args.fixed_latent_niche_within_std], dtype=np.float32
        ),
        fixed_latent_niche_normalize_radius=np.asarray(
            [not args.fixed_latent_niche_no_normalize_radius], dtype=np.int32
        ),
        fixed_latent_niche_labels=fixed_latent_niche_labels_np,
        fixed_latent_niche_within_mean_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_within_mean_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_niche_within_max_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_within_max_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_niche_between_mean_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_between_mean_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_niche_between_min_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_between_min_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_niche_center_mean_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_center_mean_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_niche_separation_margin_l2=np.asarray(
            [fixed_latent_niche_stats.get("latent_niche_separation_margin_l2", np.nan)],
            dtype=np.float32,
        ),
        fixed_latent_noise_std=np.asarray(
            [args.fixed_latent_noise_std], dtype=np.float32
        ),
        fixed_latent_noise_normalize=np.asarray(
            [not args.fixed_latent_noise_no_normalize], dtype=np.int32
        ),
        fixed_latent_uniformity_active=np.asarray([fixed_latent_uniformity_active]),
        fixed_latent_uniformity_weight=np.asarray(
            [args.fixed_latent_uniformity_weight], dtype=np.float32
        ),
        fixed_latent_uniformity_batch_size=np.asarray(
            [args.fixed_latent_uniformity_batch_size], dtype=np.int32
        ),
        fixed_latent_uniformity_t=np.asarray(
            [args.fixed_latent_uniformity_t], dtype=np.float32
        ),
        fixed_latent_uniformity_sample_mode=np.asarray(
            [args.fixed_latent_uniformity_sample_mode]
        ),
        generator_channels=np.asarray([args.generator_channels], dtype=np.int32),
        set_generator_dim=np.asarray([args.set_generator_dim], dtype=np.int32),
        set_generator_depth=np.asarray([args.set_generator_depth], dtype=np.int32),
        set_generator_heads=np.asarray([args.set_generator_heads], dtype=np.int32),
        set_generator_mlp_ratio=np.asarray(
            [args.set_generator_mlp_ratio], dtype=np.int32
        ),
        set_generator_dropout=np.asarray(
            [args.set_generator_dropout], dtype=np.float32
        ),
        set_generator_elite_context_size=np.asarray(
            [args.set_generator_elite_context_size], dtype=np.int32
        ),
        set_generator_elite_context_pool_size=np.asarray(
            [
                -1
                if args.set_generator_elite_context_pool_size is None
                else args.set_generator_elite_context_pool_size
            ],
            dtype=np.int32,
        ),
        discriminator_channels=np.asarray(
            [args.discriminator_channels], dtype=np.int32
        ),
        set_discriminator_dim=np.asarray([args.set_discriminator_dim], dtype=np.int32),
        set_discriminator_depth=np.asarray(
            [args.set_discriminator_depth], dtype=np.int32
        ),
        set_discriminator_heads=np.asarray(
            [args.set_discriminator_heads], dtype=np.int32
        ),
        set_discriminator_mlp_ratio=np.asarray(
            [args.set_discriminator_mlp_ratio], dtype=np.int32
        ),
        set_discriminator_dropout=np.asarray(
            [args.set_discriminator_dropout], dtype=np.float32
        ),
        train_on_decoded=np.asarray([args.train_on_decoded], dtype=np.int32),
        optimizer_type=np.asarray([args.optimizer_type]),
        elite_sampling=np.asarray([args.elite_sampling]),
        elite_pool_size=np.asarray(
            [
                args.elite_pool_size
                if args.elite_pool_size is not None
                else args.buffer_multiplier * args.batch_size
            ],
            dtype=np.int32,
        ),
        ranker_list_size=np.asarray([args.ranker_list_size], dtype=np.int32),
        ranker_steps=np.asarray([args.ranker_steps], dtype=np.int32),
        ranker_archive_size=np.asarray([args.ranker_archive_size], dtype=np.int32),
        ranker_generator_elite_margin=np.asarray(
            [args.ranker_generator_elite_margin], dtype=np.int32
        ),
        ranker_weight=np.asarray([args.ranker_weight], dtype=np.float32),
        ranker_target_curve=np.asarray([args.ranker_target_curve]),
        ranker_tau=np.asarray([args.ranker_tau], dtype=np.float32),
        ranker_target_scope=np.asarray([args.ranker_target_scope]),
        ranker_niche_local_targets=np.asarray(
            [args.ranker_niche_local_targets], dtype=np.int32
        ),
        ranker_list_repeats=np.asarray([args.ranker_list_repeats], dtype=np.int32),
        ranker_fake_weight=np.asarray([args.ranker_fake_weight], dtype=np.float32),
        ranker_fake_repeats=np.asarray([args.ranker_fake_repeats], dtype=np.int32),
        ranker_sample_pool_size=np.asarray(
            [
                -1
                if args.ranker_sample_pool_size is None
                else args.ranker_sample_pool_size
            ],
            dtype=np.int32,
        ),
        ranker_sample_mode=np.asarray([args.ranker_sample_mode]),
        d_score_center_weight=np.asarray(
            [args.d_score_center_weight], dtype=np.float32
        ),
        d_score_scale_weight=np.asarray([args.d_score_scale_weight], dtype=np.float32),
        d_score_target_std=np.asarray([args.d_score_target_std], dtype=np.float32),
        utility_target_scale=np.asarray([args.utility_target_scale], dtype=np.float32),
        utility_loss=np.asarray([args.utility_loss]),
        utility_weight=np.asarray([args.utility_weight], dtype=np.float32),
        generator_utility_weight=np.asarray(
            [args.generator_utility_weight], dtype=np.float32
        ),
        utility_clip=np.asarray([args.utility_clip], dtype=np.float32),
        proposal_pool_size=np.asarray(
            [-1 if args.proposal_pool_size is None else args.proposal_pool_size],
            dtype=np.int32,
        ),
        proposal_top_k=np.asarray(
            [-1 if args.proposal_top_k is None else args.proposal_top_k],
            dtype=np.int32,
        ),
        proposal_diversity_min_hamming=np.asarray(
            [args.proposal_diversity_min_hamming], dtype=np.float32
        ),
        proposal_diversity_topk_frac=np.asarray(
            [args.proposal_diversity_topk_frac], dtype=np.float32
        ),
        proposal_buffer_novelty_min_hamming=np.asarray(
            [args.proposal_buffer_novelty_min_hamming], dtype=np.float32
        ),
        proposal_buffer_novelty_reference_size=np.asarray(
            [args.proposal_buffer_novelty_reference_size], dtype=np.int32
        ),
        proposal_buffer_reject_exact_design_duplicates=np.asarray(
            [args.proposal_buffer_reject_exact_design_duplicates], dtype=np.int32
        ),
        proposal_buffer_novelty_threshold=np.asarray(
            [args.proposal_buffer_novelty_threshold], dtype=np.float32
        ),
        proposal_evolution_fraction=np.asarray(
            [args.proposal_evolution_fraction], dtype=np.float32
        ),
        proposal_evolution_parent_source=np.asarray(
            [args.proposal_evolution_parent_source]
        ),
        proposal_evolution_crossover=np.asarray([args.proposal_evolution_crossover]),
        proposal_evolution_mutation_rate=np.asarray(
            [args.proposal_evolution_mutation_rate], dtype=np.float32
        ),
        proposal_evolution_mutation_scale=np.asarray(
            [args.proposal_evolution_mutation_scale], dtype=np.float32
        ),
        proposal_gradient_steps=np.asarray(
            [args.proposal_gradient_steps], dtype=np.int32
        ),
        proposal_gradient_step_size=np.asarray(
            [args.proposal_gradient_step_size], dtype=np.float32
        ),
        proposal_gradient_mode=np.asarray([args.proposal_gradient_mode]),
        proposal_gradient_normalize=np.asarray(
            [not args.proposal_gradient_no_normalize], dtype=np.int32
        ),
        proposal_gradient_noise=np.asarray(
            [args.proposal_gradient_noise], dtype=np.float32
        ),
        proposal_gradient_keep_original=np.asarray(
            [args.proposal_gradient_keep_original], dtype=np.int32
        ),
        ga_offspring_fraction=np.asarray(
            [args.ga_offspring_fraction], dtype=np.float32
        ),
        ga_pool_size=np.asarray(
            [-1 if args.ga_pool_size is None else args.ga_pool_size],
            dtype=np.int32,
        ),
        ga_parent_pool_size=np.asarray([args.ga_parent_pool_size], dtype=np.int32),
        ga_mutation_rate=np.asarray([args.ga_mutation_rate], dtype=np.float32),
        ga_mutation_scale=np.asarray([args.ga_mutation_scale], dtype=np.float32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        buffer_diversity_min_hamming=np.asarray(
            [args.buffer_diversity_min_hamming], dtype=np.float32
        ),
        buffer_diversity_topk_frac=np.asarray(
            [args.buffer_diversity_topk_frac], dtype=np.float32
        ),
        niche_buffer_count=np.asarray([args.niche_buffer_count], dtype=np.int32),
        niche_buffer_min_hamming=np.asarray(
            [args.niche_buffer_min_hamming], dtype=np.float32
        ),
        niche_buffer_view_mode=np.asarray([args.niche_buffer_view_mode]),
        niche_buffer_cross_min_hamming=np.asarray(
            [args.niche_buffer_cross_min_hamming], dtype=np.float32
        ),
        niche_buffer_cross_reference_top_k=np.asarray(
            [args.niche_buffer_cross_reference_top_k], dtype=np.int32
        ),
        niche_output_separation_weight=np.asarray(
            [args.niche_output_separation_weight], dtype=np.float32
        ),
        niche_output_separation_margin=np.asarray(
            [args.niche_output_separation_margin], dtype=np.float32
        ),
        density_filter_radius=np.asarray([args.density_filter_radius], dtype=np.int32),
        density_filter_warmup_iters=np.asarray(
            [args.density_filter_warmup_iters], dtype=np.int32
        ),
        density_filter_final_radius=np.asarray(
            [args.density_filter_final_radius], dtype=np.int32
        ),
        density_filter_active_radius=np.asarray(
            [evaluator.config.density_filter_radius], dtype=np.int32
        ),
        density_filter_switched=np.asarray([density_filter_switched], dtype=np.int32),
        density_filter_extra_eval_count=np.asarray(
            [density_filter_extra_eval_count], dtype=np.int32
        ),
        projection_beta=np.asarray([args.projection_beta], dtype=np.float32),
        projection_eta=np.asarray([args.projection_eta], dtype=np.float32),
        binhead_connect_support=np.asarray(
            [args.binhead_connect_support], dtype=np.int32
        ),
        volume_ladder=np.asarray(args.volume_ladder, dtype=np.float32),
        compliance_ladder=np.asarray(args.compliance_ladder, dtype=np.float32),
        roughness_ladder=np.asarray(args.roughness_ladder, dtype=np.float32),
        connectivity_ladder=np.asarray(args.connectivity_ladder, dtype=np.float32),
        diversity_ladder=np.asarray(args.diversity_ladder, dtype=np.float32),
        diversity_reference_size=np.asarray(
            [args.diversity_reference_size],
            dtype=np.int32,
        ),
        diversity_chamfer_max_points=np.asarray(
            [args.diversity_chamfer_max_points],
            dtype=np.int32,
        ),
        removal_ladder_volumes=np.asarray(
            args.removal_ladder_volumes,
            dtype=np.float32,
        ),
        removal_ladder_compliances=np.asarray(
            args.removal_ladder_compliances,
            dtype=np.float32,
        ),
        removal_ladder_connectivity_max=np.asarray(
            [
                float("nan")
                if args.removal_ladder_connectivity_max is None
                else args.removal_ladder_connectivity_max
            ],
            dtype=np.float32,
        ),
        connectivity_max=np.asarray(
            [float("nan") if args.connectivity_max is None else args.connectivity_max],
            dtype=np.float32,
        ),
        ladder_sequence=np.asarray(
            [f"{kind}:{bound:g}" for kind, bound in ladder_sequence]
        ),
        levels_ladder=np.asarray(args.levels_ladder),
        levels_ladder_objectives=np.asarray(
            [rung.name for rung in levels_ladder_rungs]
        ),
        levels_ladder_final_open=np.asarray([args.levels_ladder_final_open]),
        residual_scale=np.asarray([args.residual_scale], dtype=np.float32),
        load_case=np.asarray([args.load_case]),
        robust_load_cases=np.asarray(args.robust_load_cases),
        robust_load_aggregate=np.asarray([args.robust_load_aggregate]),
        robust_load_cvar_frac=np.asarray(
            [args.robust_load_cvar_frac],
            dtype=np.float32,
        ),
        load_scale=np.asarray([args.load_scale], dtype=np.float32),
        fem_workers=np.asarray([args.fem_workers], dtype=np.int32),
        compliance_solver=np.asarray([args.compliance_solver]),
        matrix_free_cg_max_iter=np.asarray(
            [args.matrix_free_cg_max_iter],
            dtype=np.int32,
        ),
        matrix_free_cg_tol=np.asarray([args.matrix_free_cg_tol], dtype=np.float32),
        matrix_free_cg_device=np.asarray([args.matrix_free_cg_device]),
        matrix_free_cg_dtype=np.asarray([args.matrix_free_cg_dtype]),
        e_max=np.asarray([args.e_max], dtype=np.float32),
        e_min=np.asarray([args.e_min], dtype=np.float32),
        poisson_ratio=np.asarray([args.poisson_ratio], dtype=np.float32),
        hard_binarize=np.asarray([args.hard_binarize], dtype=np.int32),
        generator_hidden_dims=np.asarray(args.generator_hidden_dims, dtype=np.int32),
        discriminator_hidden_dims=np.asarray(
            args.discriminator_hidden_dims, dtype=np.int32
        ),
        g_torch_optimizer=np.asarray([args.g_torch_optimizer]),
        d_torch_optimizer=np.asarray([args.d_torch_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        g_momentum=np.asarray([args.g_momentum], dtype=np.float32),
        d_momentum=np.asarray([args.d_momentum], dtype=np.float32),
        discriminator_steps=np.asarray([args.discriminator_steps], dtype=np.int32),
        weight_clip=np.asarray([args.weight_clip], dtype=np.float32),
        gradient_penalty_weight=np.asarray(
            [args.gradient_penalty_weight], dtype=np.float32
        ),
    )

    return {
        "archive_best_value": metrics["archive_best_value"],
        "archive_best_compliance": metrics["archive_best_compliance"],
        "best_feasible_compliance": metrics["best_feasible_compliance"],
        "best_any_compliance": metrics["best_any_compliance"],
        "feasible_count": metrics["feasible_count"],
        "feasible_rate": metrics["feasible_rate"],
        "mean_compliance_topk": float(actual_compliance.mean()),
        "best_relative_compliance": float(relative_compliance[0]),
        "mean_relative_compliance_topk": float(relative_compliance.mean()),
        "solid_compliance": float(evaluator.solid_compliance),
        "mean_l2": mean_l2,
        "mean_hamming": mean_hamming,
        "curiosity": args.curiosity,
        "curiosity_space": args.curiosity_space,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "artifact_path": str(args.output_dir / f"top_designs_{suffix}.npz"),
        "history_plot": "" if args.no_history_plot else str(history_plot_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(args)
    logger.info(f"Saved outputs to {result['output_dir']}")
    logger.info(f"Saved raw artifacts to {result['artifact_path']}")
    logger.info(f"Archive-best value vector: {result['archive_best_value']}")
    logger.info(
        f"Top-k summary: archive_best_compliance={result['archive_best_compliance']:.4f} "
        f"best_feasible_compliance={result['best_feasible_compliance']:.4f} "
        f"best_any_compliance={result['best_any_compliance']:.4f} "
        f"feasible_count={result['feasible_count']} feasible_rate={result['feasible_rate']:.3f} "
        f"best_relative_compliance={result['best_relative_compliance']:.4f} "
        f"mean_compliance_topk={result['mean_compliance_topk']:.4f} "
        f"mean_relative_compliance_topk={result['mean_relative_compliance_topk']:.4f} "
        f"solid_compliance={result['solid_compliance']:.4f} "
        f"mean_l2={result['mean_l2']:.4f} mean_hamming={result['mean_hamming']:.4f}"
    )


if __name__ == "__main__":
    main()
