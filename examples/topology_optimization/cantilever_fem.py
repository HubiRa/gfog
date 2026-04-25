import argparse
import math
import random
import sys
import types
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
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels
from gfog.curiosity import WarmupCosine, WangIsolaUniformity, WangIsolaUniformityConfig
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
from gfog.opt.latents_sampler import LatentSamplerLambda
from gfog.utils import uniformity_loss


EncodingMode = Literal[
    "direct",
    "coarse",
    "binary_coarse",
    "topk_volume",
    "coarse_topk_volume",
    "coarse_residual",
    "soft_volume",
]


LadderKind = Literal["volume", "compliance", "roughness"]


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


class ConvDiscriminator(nn.Module):
    """Small spectral-normalized CNN discriminator for flattened grids."""

    def __init__(
        self,
        input_height: int,
        input_width: int,
        channels: int = 32,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda layer: layer)
        self.net = nn.Sequential(
            sn(nn.Conv2d(1, channels, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            sn(
                nn.Conv2d(
                    channels * 2, channels * 4, kernel_size=3, stride=2, padding=1
                )
            ),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(channels * 4, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], 1, self.input_height, self.input_width)
        return self.net(x)


@dataclass
class FEMConfig:
    grid_width: int = 32
    grid_height: int = 16
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
    volume_ladder: tuple[float, ...] = ()
    compliance_ladder: tuple[float, ...] = ()
    roughness_ladder: tuple[float, ...] = ()
    ladder_sequence: tuple[tuple[LadderKind, float], ...] = ()
    load_scale: float = 1.0
    density_filter_radius: int = 1
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    hard_binarize: bool = False


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
        if kind not in {"volume", "compliance", "roughness"}:
            raise ValueError(
                f"Invalid ladder kind '{kind_raw}'. Expected one of volume, compliance, roughness"
            )
        parsed.append((kind, float(value_raw)))
    return tuple(parsed)


def get_level_names(
    volume_ladder: list[float],
    compliance_ladder: list[float],
    roughness_ladder: list[float],
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
        self.nelx = config.grid_width
        self.nely = config.grid_height
        self.code_width = config.coarse_grid_width or config.grid_width
        self.code_height = config.coarse_grid_height or config.grid_height
        self.code_dim = self.code_width * self.code_height
        self.nelems = self.nelx * self.nely
        self.nnodes = (self.nelx + 1) * (self.nely + 1)
        self.ndof = 2 * self.nnodes
        self.ke = self._element_stiffness(config.poisson_ratio)
        self.edof_mat = self._build_edof_mat()
        self.iK = np.kron(self.edof_mat, np.ones((8, 1), dtype=np.int32)).ravel()
        self.jK = np.kron(self.edof_mat, np.ones((1, 8), dtype=np.int32)).ravel()
        self.fixed_dofs, self.free_dofs = self._build_boundary_conditions()
        self.force = self._build_force_vector()
        self.filter_offsets, self.filter_weights = self._build_filter_kernel()
        self.solid_compliance = self._compute_solid_compliance()

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

    def _build_force_vector(self) -> np.ndarray:
        force = np.zeros(self.ndof, dtype=np.float64)
        load_row = self.nely // 2
        load_node = load_row * (self.nelx + 1) + self.nelx
        force[2 * load_node + 1] = -self.config.load_scale
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

    def _solve_compliance(self, density_phys: np.ndarray) -> float:
        K = self._assemble_stiffness(density_phys)
        K_ff = K[self.free_dofs][:, self.free_dofs]
        u_f = spla.spsolve(K_ff, self.force[self.free_dofs])
        compliance = float(self.force[self.free_dofs] @ u_f)
        return compliance

    def decode_designs_numpy(self, x_np: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x_np, dtype=np.float32)

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
            k = int(round(self.config.volume_max * self.nely * self.nelx))
            x_phys = np.stack(
                [
                    project_scores_to_topk_binary_numpy(sample, k)
                    for sample in final_scores
                ],
                axis=0,
            ).reshape(-1, self.nely, self.nelx)
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
        else:
            raise ValueError(f"Unknown encoding mode: {self.config.encoding}")

        decoded = []
        for sample_raw in x_phys.reshape(-1, self.nely, self.nelx):
            sample = self._apply_density_filter(
                sample_raw.astype(np.float64, copy=False)
            )
            sample = apply_projection_numpy(
                sample,
                beta=self.config.projection_beta,
                eta=self.config.projection_eta,
                hard_binarize=self.config.hard_binarize,
            )
            decoded.append(sample)
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

    def evaluate_densities_numpy(self, x_phys: np.ndarray) -> list[list[float]]:
        results: list[list[float]] = []
        for sample in np.asarray(x_phys, dtype=np.float64).reshape(
            -1, self.nely, self.nelx
        ):
            volume = float(sample.mean())
            dx = float(np.abs(sample[:, 1:] - sample[:, :-1]).mean())
            dy = float(np.abs(sample[1:, :] - sample[:-1, :]).mean())
            roughness = 0.5 * (dx + dy)
            compliance = self._solve_compliance(sample)
            level_values: list[float] = []
            if self.config.ladder_sequence:
                for kind, bound in self.config.ladder_sequence:
                    if kind == "volume":
                        level_values.append(max(volume - bound, 0.0))
                    elif kind == "compliance":
                        level_values.append(max(compliance - bound, 0.0))
                    elif kind == "roughness":
                        level_values.append(max(roughness - bound, 0.0))
                    else:
                        raise ValueError(f"Unknown ladder kind: {kind}")
            else:
                for bound in self.config.volume_ladder:
                    level_values.append(max(volume - bound, 0.0))
                for bound in self.config.compliance_ladder:
                    level_values.append(max(compliance - bound, 0.0))
                for bound in self.config.roughness_ladder:
                    level_values.append(max(roughness - bound, 0.0))
            level_values.extend(
                [
                    max(volume - self.config.volume_max, 0.0),
                    max(roughness - self.config.roughness_max, 0.0),
                    compliance,
                ]
            )
            results.append(level_values)
        return results

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[list[float]]:
        if isinstance(theta, torch.Tensor):
            x_np = theta.detach().to("cpu", torch.float32).numpy()
        else:
            x_np = np.asarray(theta, dtype=np.float32)

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
        self.code_width = config.coarse_grid_width or config.grid_width
        self.code_height = config.coarse_grid_height or config.grid_height
        self.code_dim = self.code_width * self.code_height
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


def lexicographic_order(values: list[list[float]]) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: tuple(values[idx]))


def plackett_luce_loss(scores_best_to_worst: torch.Tensor) -> torch.Tensor:
    scores = scores_best_to_worst.reshape(-1)
    if scores.numel() < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    log_denoms = torch.logcumsumexp(scores.flip(0), dim=0).flip(0)
    return -(scores - log_denoms).mean()


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
    ) -> None:
        self.ranker_list_size = ranker_list_size
        self.ranker_steps = ranker_steps
        self.generator_elite_margin = generator_elite_margin
        super().__init__(components)

    def _ranked_buffer_subset(self, k: int) -> torch.Tensor:
        current_len = len(self.buffer.B)
        if current_len == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        k = min(k, current_len)
        if k == current_len:
            ranked = self.buffer.B.get_top_k(k)
        else:
            # Buffer indices are already sorted best-to-worst, so sampled sorted
            # positions preserve the true lexicographic order.
            positions = torch.randperm(current_len)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.B.get(int(pos)) for pos in positions])
        return ranked.to(self.gan.device, self.gan.dtype)

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
        proposals = self.gan.G(z)
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
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))


class RankedLSGANOpt(BufferPlackettLuceRankerOpt):
    """LSGAN with an additional Plackett-Luce ranking loss on buffer elites."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int = 32,
        ranker_steps: int = 1,
        ranker_weight: float = 0.1,
    ) -> None:
        super().__init__(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            generator_elite_margin=False,
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
            fake = self.gan.G(z)
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
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = 0.5 * ((scores - 1.0) ** 2).mean()
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals


def make_optimizer(
    optimizer_type: str,
    opt_components: components.OptComponents,
    *,
    archive_size: int | None = None,
    ranker_list_size: int = 32,
    ranker_steps: int = 1,
    ranker_generator_elite_margin: bool = False,
    ranker_weight: float = 0.1,
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
        )
    if optimizer_type == "ranked_lsgan":
        return RankedLSGANOpt(
            opt_components,
            ranker_list_size=ranker_list_size,
            ranker_steps=ranker_steps,
            ranker_weight=ranker_weight,
        )
    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def make_decoded_density_optimizer(
    optimizer_type: str,
    opt_components: components.OptComponents,
    *,
    evaluator: FEMCantileverEvaluator,
) -> BaseOpt:
    kwargs = {
        "decoder": evaluator.decode_designs_torch,
        "density_objective": evaluator.evaluate_densities_numpy,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GFog FEM cantilever benchmark")
    parser.add_argument("--grid_width", type=int, default=32)
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--backend", choices=["scipy", "torchfem"], default="scipy")
    parser.add_argument("--torchfem_src", type=str, default=None)
    parser.add_argument("--torchfem_device", type=str, default="cpu")
    parser.add_argument("--n_iter", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--encoding",
        choices=[
            "direct",
            "coarse",
            "binary_coarse",
            "topk_volume",
            "coarse_topk_volume",
            "coarse_residual",
            "soft_volume",
        ],
        default="direct",
    )
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--coarse_grid_width", type=int, default=None)
    parser.add_argument("--coarse_grid_height", type=int, default=None)
    parser.add_argument("--generator_type", choices=["mlp", "conv"], default="mlp")
    parser.add_argument("--discriminator_type", choices=["mlp", "conv"], default="mlp")
    parser.add_argument("--generator_channels", type=int, default=64)
    parser.add_argument("--discriminator_channels", type=int, default=32)
    parser.add_argument(
        "--generator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument(
        "--discriminator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--curiosity", type=float, default=10.0)
    parser.add_argument(
        "--curiosity_space",
        choices=["raw", "topology"],
        default="raw",
        help="Apply curiosity to raw generator outputs or decoded topology fields.",
    )
    parser.add_argument(
        "--curiosity_schedule",
        choices=["none", "warmup_cosine"],
        default="none",
        help="Optional schedule multiplier for curiosity weight.",
    )
    parser.add_argument("--curiosity_warmup_frac", type=float, default=0.05)
    parser.add_argument("--curiosity_min", type=float, default=0.0)
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
            "ranked_lsgan",
        ],
        default="default",
    )
    parser.add_argument("--ranker_list_size", type=int, default=32)
    parser.add_argument("--ranker_steps", type=int, default=1)
    parser.add_argument("--ranker_archive_size", type=int, default=8192)
    parser.add_argument("--ranker_generator_elite_margin", action="store_true")
    parser.add_argument(
        "--ranker_weight",
        type=float,
        default=0.1,
        help="Weight for auxiliary PL rank loss in ranked_lsgan.",
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
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--roughness_max", type=float, default=0.18)
    parser.add_argument("--volume_ladder", nargs="*", type=float, default=[])
    parser.add_argument("--compliance_ladder", nargs="*", type=float, default=[])
    parser.add_argument("--roughness_ladder", nargs="*", type=float, default=[])
    parser.add_argument(
        "--ladder_sequence",
        nargs="*",
        type=str,
        default=[],
        help="Explicit interleaved ladder sequence like volume:0.52 compliance:318 volume:0.50 compliance:314",
    )
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--e_max", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.3)
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument("--projection_beta", type=float, default=0.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/fem_cantilever"),
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train_on_decoded:
        if args.encoding not in {"direct", "soft_volume", "coarse", "coarse_residual"}:
            raise ValueError(
                "--train_on_decoded only supports direct, soft_volume, coarse, and coarse_residual encodings"
            )
        if args.hard_binarize:
            raise ValueError("--train_on_decoded does not support --hard_binarize")

    ladder_sequence = parse_ladder_sequence(args.ladder_sequence)

    cfg = FEMConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
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
        volume_ladder=tuple(args.volume_ladder),
        compliance_ladder=tuple(args.compliance_ladder),
        roughness_ladder=tuple(args.roughness_ladder),
        ladder_sequence=ladder_sequence,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
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
    if args.encoding in {"direct", "soft_volume", "topk_volume"}:
        code_height = args.grid_height
        code_width = args.grid_width
    coarse_dim = code_height * code_width
    full_dim = args.grid_height * args.grid_width
    f_dim = coarse_dim + full_dim if args.encoding == "coarse_residual" else coarse_dim
    d_input_dim = full_dim if args.train_on_decoded else f_dim
    device = torch.device("cpu")

    fn = components.Fn(f=evaluator, input_dim=f_dim, device=device, dtype=torch.float32)
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
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")

    if args.discriminator_type == "mlp":
        d = MLP(
            input_dim=d_input_dim,
            output_dim=1,
            hidden_dims=args.discriminator_hidden_dims,
            use_spectral_norm=True,
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
            use_spectral_norm=True,
        ).to(device)
    else:
        raise ValueError(f"Unknown discriminator_type: {args.discriminator_type}")

    level_names = get_level_names(
        args.volume_ladder,
        args.compliance_ladder,
        args.roughness_ladder,
        ladder_sequence,
    )
    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=Levels(level_names),
        )
    )

    curiosity_scheduler = None
    if args.curiosity > 0 and args.curiosity_schedule == "warmup_cosine":
        curiosity_scheduler = WarmupCosine(
            total_steps=args.n_iter,
            warmup_frac=args.curiosity_warmup_frac,
            base=1.0,
            min_val=args.curiosity_min,
        )
    elif args.curiosity_schedule != "none":
        raise ValueError(f"Unknown curiosity_schedule: {args.curiosity_schedule}")

    curiosity_loss = None
    if args.curiosity > 0:
        if args.curiosity_space == "raw":
            curiosity_loss = WangIsolaUniformity(
                WangIsolaUniformityConfig(use_buffer=True, weight=args.curiosity),
                buffer=buffer.B,
                scheduler=curiosity_scheduler,
            )
        elif args.curiosity_space == "topology":
            curiosity_loss = TopologySpaceUniformity(
                evaluator=evaluator,
                buffer=buffer.B,
                weight=args.curiosity,
                use_buffer=True,
                scheduler=curiosity_scheduler,
            )
        else:
            raise ValueError(f"Unknown curiosity_space: {args.curiosity_space}")

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
            lambda b, d: torch.randn(b, d), b=args.batch_size, d=args.latent_dim
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
        weight_clip=args.weight_clip if args.optimizer_type == "wgan" else None,
        gradient_penalty_weight=args.gradient_penalty_weight,
    )
    if args.train_on_decoded:
        if args.optimizer_type == "plackett_luce":
            raise ValueError(
                "--optimizer_type plackett_luce does not support --train_on_decoded yet"
            )
        optimizer = make_decoded_density_optimizer(
            args.optimizer_type,
            opt_components,
            evaluator=evaluator,
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
        )

    logger.info(
        f"FEMCantilever: grid={args.grid_width}x{args.grid_height} code_grid={code_width}x{code_height} backend={args.backend} encoding={args.encoding} optimizer={args.optimizer_type} n_iter={args.n_iter} "
        f"G={args.generator_type} D={args.discriminator_type} "
        f"curiosity={args.curiosity} curiosity_space={args.curiosity_space} curiosity_schedule={args.curiosity_schedule} g_opt={args.g_torch_optimizer} d_opt={args.d_torch_optimizer} g_lr={args.g_lr} d_lr={args.d_lr} ranker_list_size={args.ranker_list_size} ranker_steps={args.ranker_steps} ranker_weight={args.ranker_weight} filter_radius={args.density_filter_radius} residual_scale={args.residual_scale} "
        f"projection_beta={args.projection_beta} hard_binarize={args.hard_binarize} train_on_decoded={args.train_on_decoded}"
    )
    optimizer.optimize(args.n_iter, verbose=True)

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
    top_values = np.asarray(buffer.B.get_sorted_values()[:top_k], dtype=np.float32)
    summary_values = np.asarray(buffer.B.get_sorted_values(), dtype=np.float32)
    metrics = compliance_summary(summary_values)
    actual_compliance = top_values[:, -1].copy()
    relative_compliance = actual_compliance / evaluator.solid_compliance
    volume_violation = np.maximum(
        top_designs.cpu().numpy().mean(axis=(1, 2)) - args.volume_max, 0.0
    ).astype(np.float32)
    top_designs_np = top_designs.cpu().numpy()
    dx = np.abs(top_designs_np[:, :, 1:] - top_designs_np[:, :, :-1]).mean(axis=(1, 2))
    dy = np.abs(top_designs_np[:, 1:, :] - top_designs_np[:, :-1, :]).mean(axis=(1, 2))
    roughness_violation = np.maximum(0.5 * (dx + dy) - args.roughness_max, 0.0).astype(
        np.float32
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"curiosity_{args.curiosity:g}_seed_{args.seed}"
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
    np.savez_compressed(
        args.output_dir / f"top_designs_{suffix}.npz",
        designs=top_designs.cpu().numpy(),
        archive_tensors=top_archive_tensors.cpu().numpy(),
        raw_design_code=raw_design_code,
        values=top_values,
        all_archive_values=summary_values,
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
        solid_compliance=np.asarray([evaluator.solid_compliance], dtype=np.float32),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        curiosity_space=np.asarray([args.curiosity_space]),
        curiosity_schedule=np.asarray([args.curiosity_schedule]),
        curiosity_warmup_frac=np.asarray(
            [args.curiosity_warmup_frac], dtype=np.float32
        ),
        curiosity_min=np.asarray([args.curiosity_min], dtype=np.float32),
        seed=np.asarray([args.seed], dtype=np.int32),
        grid_width=np.asarray([args.grid_width], dtype=np.int32),
        grid_height=np.asarray([args.grid_height], dtype=np.int32),
        coarse_grid_width=np.asarray([code_width], dtype=np.int32),
        coarse_grid_height=np.asarray([code_height], dtype=np.int32),
        backend=np.asarray([args.backend]),
        encoding=np.asarray([args.encoding]),
        generator_type=np.asarray([args.generator_type]),
        discriminator_type=np.asarray([args.discriminator_type]),
        generator_channels=np.asarray([args.generator_channels], dtype=np.int32),
        discriminator_channels=np.asarray(
            [args.discriminator_channels], dtype=np.int32
        ),
        train_on_decoded=np.asarray([args.train_on_decoded], dtype=np.int32),
        optimizer_type=np.asarray([args.optimizer_type]),
        ranker_list_size=np.asarray([args.ranker_list_size], dtype=np.int32),
        ranker_steps=np.asarray([args.ranker_steps], dtype=np.int32),
        ranker_archive_size=np.asarray([args.ranker_archive_size], dtype=np.int32),
        ranker_generator_elite_margin=np.asarray(
            [args.ranker_generator_elite_margin], dtype=np.int32
        ),
        ranker_weight=np.asarray([args.ranker_weight], dtype=np.float32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        density_filter_radius=np.asarray([args.density_filter_radius], dtype=np.int32),
        projection_beta=np.asarray([args.projection_beta], dtype=np.float32),
        projection_eta=np.asarray([args.projection_eta], dtype=np.float32),
        volume_ladder=np.asarray(args.volume_ladder, dtype=np.float32),
        compliance_ladder=np.asarray(args.compliance_ladder, dtype=np.float32),
        roughness_ladder=np.asarray(args.roughness_ladder, dtype=np.float32),
        ladder_sequence=np.asarray(
            [f"{kind}:{bound:g}" for kind, bound in ladder_sequence]
        ),
        residual_scale=np.asarray([args.residual_scale], dtype=np.float32),
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
        "mean_l2": pairwise_l2_mean(top_designs),
        "mean_hamming": pairwise_hamming_mean(top_designs),
        "curiosity": args.curiosity,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "artifact_path": str(args.output_dir / f"top_designs_{suffix}.npz"),
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
