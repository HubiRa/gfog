#!/usr/bin/env python
"""Compare direct sparse FEM compliance with matrix-free batched CG."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cantilever_fem import (
    FEMCantileverEvaluator,
    FEMConfig,
    matrix_free_cg_torch_dtype,
)


@dataclass
class BatchedCGResult:
    compliances: np.ndarray
    relative_residuals: np.ndarray
    iterations: np.ndarray
    elapsed_sec: float


class MatrixFreeElasticityCG:
    """Forward-only batched compliance solver for regular-grid FEM designs."""

    def __init__(
        self,
        evaluator: FEMCantileverEvaluator,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.evaluator = evaluator
        self.device = torch.device(device)
        self.dtype = dtype
        free_index = np.full(evaluator.ndof, -1, dtype=np.int64)
        free_index[evaluator.free_dofs] = np.arange(evaluator.n_free_dofs)
        self.edof = torch.as_tensor(
            evaluator.edof_mat.astype(np.int64),
            device=self.device,
        )
        self.free_dofs = torch.as_tensor(
            evaluator.free_dofs.astype(np.int64),
            device=self.device,
        )
        self.ke = torch.as_tensor(
            evaluator.ke,
            device=self.device,
            dtype=self.dtype,
        )
        self.force_free = torch.as_tensor(
            evaluator.force_free,
            device=self.device,
            dtype=self.dtype,
        )
        self.ndof = evaluator.ndof
        self.n_free_dofs = evaluator.n_free_dofs
        self.e_min = float(evaluator.config.e_min)
        self.e_max = float(evaluator.config.e_max)
        self.simp_p = float(evaluator.config.simp_p)

    def _penalized_moduli(self, densities: torch.Tensor) -> torch.Tensor:
        flat = densities.reshape(densities.shape[0], -1)
        return self.e_min + flat.pow(self.simp_p) * (self.e_max - self.e_min)

    def _matvec(self, x_free: torch.Tensor, moduli: torch.Tensor) -> torch.Tensor:
        batch_size = x_free.shape[0]
        x_global = torch.zeros(
            (batch_size, self.ndof),
            device=self.device,
            dtype=self.dtype,
        )
        x_global[:, self.free_dofs] = x_free
        x_elem = x_global[:, self.edof]
        kx_elem = torch.einsum("ij,bej->bei", self.ke, x_elem)
        kx_elem = kx_elem * moduli[:, :, None]
        out_global = torch.zeros_like(x_global)
        scatter_index = self.edof.reshape(1, -1).expand(batch_size, -1)
        out_global.scatter_add_(1, scatter_index, kx_elem.reshape(batch_size, -1))
        return out_global[:, self.free_dofs]

    def _jacobi_diagonal(self, moduli: torch.Tensor) -> torch.Tensor:
        batch_size = moduli.shape[0]
        elem_diag = torch.diagonal(self.ke).reshape(1, 1, -1) * moduli[:, :, None]
        diag_global = torch.zeros(
            (batch_size, self.ndof),
            device=self.device,
            dtype=self.dtype,
        )
        scatter_index = self.edof.reshape(1, -1).expand(batch_size, -1)
        diag_global.scatter_add_(1, scatter_index, elem_diag.reshape(batch_size, -1))
        return torch.clamp(diag_global[:, self.free_dofs], min=1e-30)

    @torch.no_grad()
    def solve(
        self,
        densities_np: np.ndarray,
        *,
        max_iter: int,
        tol: float,
        use_jacobi: bool = True,
    ) -> BatchedCGResult:
        densities = torch.as_tensor(
            np.asarray(densities_np, dtype=np.float64),
            device=self.device,
            dtype=self.dtype,
        ).reshape(-1, self.evaluator.nely, self.evaluator.nelx)
        batch_size = densities.shape[0]
        moduli = self._penalized_moduli(densities)
        b = self.force_free.reshape(1, -1).expand(batch_size, -1)
        b_norm = torch.clamp(torch.linalg.vector_norm(b, dim=1), min=1e-30)
        x = torch.zeros_like(b)
        r = b.clone()
        diag = self._jacobi_diagonal(moduli) if use_jacobi else None
        z = r / diag if diag is not None else r.clone()
        p = z.clone()
        rz = torch.sum(r * z, dim=1)
        rel = torch.linalg.vector_norm(r, dim=1) / b_norm
        iterations = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
        active = rel > tol
        start = time.perf_counter()

        for step in range(1, max_iter + 1):
            if not bool(torch.any(active).item()):
                break
            if p.device.type == "mps":
                ap = self._matvec(p, moduli)
            else:
                active_idx = torch.nonzero(active, as_tuple=False).flatten()
                ap = torch.zeros_like(p)
                ap_active = self._matvec(
                    p.index_select(0, active_idx),
                    moduli.index_select(0, active_idx),
                )
                ap.index_copy_(0, active_idx, ap_active)
            denom = torch.clamp(torch.sum(p * ap, dim=1), min=1e-30)
            alpha = rz / denom
            x_next = x + alpha[:, None] * p
            r_next = r - alpha[:, None] * ap
            z_next = r_next / diag if diag is not None else r_next
            rz_next = torch.sum(r_next * z_next, dim=1)
            beta = rz_next / torch.clamp(rz, min=1e-30)
            p_next = z_next + beta[:, None] * p

            active_col = active[:, None]
            x = torch.where(active_col, x_next, x)
            r = torch.where(active_col, r_next, r)
            z = torch.where(active_col, z_next, z)
            p = torch.where(active_col, p_next, p)
            rz = torch.where(active, rz_next, rz)
            rel = torch.linalg.vector_norm(r, dim=1) / b_norm
            newly_converged = active & (rel <= tol)
            iterations = torch.where(
                newly_converged,
                torch.full_like(iterations, step),
                iterations,
            )
            active = active & (rel > tol)

        elapsed = time.perf_counter() - start
        iterations = torch.where(
            iterations == 0,
            torch.full_like(iterations, max_iter),
            iterations,
        )
        compliances = torch.sum(b * x, dim=1)
        return BatchedCGResult(
            compliances=compliances.cpu().numpy(),
            relative_residuals=rel.cpu().numpy(),
            iterations=iterations.cpu().numpy(),
            elapsed_sec=elapsed,
        )


def _scalar(archive: np.lib.npyio.NpzFile, key: str, default: Any) -> Any:
    if key not in archive:
        return default
    value = np.ravel(archive[key])[0]
    if isinstance(default, str):
        return str(value)
    if isinstance(default, int):
        return int(value)
    return float(value)


def evaluator_from_artifact(
    artifact: Path,
    *,
    e_min_ratio: float | None = None,
) -> FEMCantileverEvaluator:
    data = np.load(artifact, allow_pickle=True)
    e_max = _scalar(data, "e_max", 1.0)
    e_min = _scalar(data, "e_min", 1e-3)
    if e_min_ratio is not None:
        e_min = e_min_ratio * e_max
    config = FEMConfig(
        grid_width=_scalar(data, "grid_width", 40),
        grid_height=_scalar(data, "grid_height", 20),
        domain_width=_scalar(data, "domain_width", 1.0),
        domain_height=_scalar(data, "domain_height", 1.0),
        volume_max=_scalar(data, "volume_max", 0.48),
        e_max=e_max,
        e_min=e_min,
        poisson_ratio=_scalar(data, "poisson_ratio", 0.3),
        load_scale=_scalar(data, "load_scale", 1.0),
        load_case=_scalar(data, "load_case", "center_point"),
        density_filter_radius=0,
        projection_beta=0.0,
    )
    return FEMCantileverEvaluator(config)


def load_designs(artifact: Path, top_k: int) -> np.ndarray:
    data = np.load(artifact, allow_pickle=True)
    if "designs" not in data:
        raise ValueError(f"{artifact} does not contain a 'designs' array")
    designs = np.asarray(data["designs"], dtype=np.float64)
    if designs.ndim != 3:
        raise ValueError(f"Expected designs with shape (n,h,w), got {designs.shape}")
    return designs[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Torch dtype for matrix-free CG. Use float32 for MPS.",
    )
    parser.add_argument(
        "--e_min_ratio",
        type=float,
        default=None,
        help="Override e_min as this fraction of e_max for both direct and CG.",
    )
    parser.add_argument("--no_jacobi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = evaluator_from_artifact(args.artifact, e_min_ratio=args.e_min_ratio)
    designs = load_designs(args.artifact, args.top_k)

    direct_start = time.perf_counter()
    direct = np.asarray(
        [evaluator._solve_compliance(design) for design in designs],
        dtype=np.float64,
    )
    direct_elapsed = time.perf_counter() - direct_start

    if args.device.startswith("mps") and args.dtype == "float64":
        raise ValueError("--device mps requires --dtype float32")
    solver = MatrixFreeElasticityCG(
        evaluator,
        device=args.device,
        dtype=matrix_free_cg_torch_dtype(args.dtype),
    )
    result = solver.solve(
        designs,
        max_iter=args.max_iter,
        tol=args.tol,
        use_jacobi=not args.no_jacobi,
    )
    abs_err = np.abs(result.compliances - direct)
    rel_err = abs_err / np.maximum(np.abs(direct), 1e-30)

    print(f"artifact={args.artifact}")
    print(
        "grid="
        f"{evaluator.nelx}x{evaluator.nely} "
        f"free_dofs={evaluator.n_free_dofs} "
        f"batch={len(designs)} "
        f"e_min/e_max={evaluator.config.e_min / evaluator.config.e_max:g} "
        f"device={args.device} dtype={args.dtype}"
    )
    print(
        f"direct_elapsed_sec={direct_elapsed:.4f} "
        f"cg_elapsed_sec={result.elapsed_sec:.4f}"
    )
    print("idx direct cg abs_err rel_err rel_res iterations")
    for idx, values in enumerate(
        zip(
            direct,
            result.compliances,
            abs_err,
            rel_err,
            result.relative_residuals,
            result.iterations,
            strict=True,
        )
    ):
        direct_i, cg_i, abs_i, rel_i, residual_i, iteration_i = values
        print(
            f"{idx:02d} "
            f"{direct_i:.12g} "
            f"{cg_i:.12g} "
            f"{abs_i:.3e} "
            f"{rel_i:.3e} "
            f"{residual_i:.3e} "
            f"{int(iteration_i)}"
        )


if __name__ == "__main__":
    main()
