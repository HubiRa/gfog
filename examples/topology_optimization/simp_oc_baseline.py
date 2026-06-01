"""Classical SIMP + optimality-criteria baseline for the FEM cantilever."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cantilever_fem import FEMCantileverEvaluator, FEMConfig


def solve_displacement(
    evaluator: FEMCantileverEvaluator,
    density: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve FEM and return full displacement, compliance, and element energies."""
    penalized = evaluator.config.e_min + (
        density.ravel(order="C") ** evaluator.config.simp_p
    ) * (evaluator.config.e_max - evaluator.config.e_min)
    s_k = evaluator.free_entry_ke * penalized[evaluator.free_entry_elements]
    k_ff = sp.coo_matrix(
        (s_k, (evaluator.iK_free, evaluator.jK_free)),
        shape=(evaluator.n_free_dofs, evaluator.n_free_dofs),
    ).tocsc()
    u_free = spla.spsolve(k_ff, evaluator.force_free)
    displacement = np.zeros(evaluator.ndof, dtype=np.float64)
    displacement[evaluator.free_dofs] = u_free
    compliance = float(evaluator.force_free @ u_free)
    u_elem = displacement[evaluator.edof_mat]
    element_energy = np.einsum("ei,ij,ej->e", u_elem, evaluator.ke, u_elem)
    return (
        displacement,
        compliance,
        element_energy.reshape(evaluator.nely, evaluator.nelx),
    )


def oc_update(
    density: np.ndarray,
    dc: np.ndarray,
    *,
    volume_fraction: float,
    move: float,
    min_density: float,
) -> np.ndarray:
    """Perform the standard OC bisection update under a volume constraint."""
    lo = 0.0
    hi = 1e9
    updated = density.copy()
    dv = np.ones_like(density)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        scale = np.sqrt(np.maximum(0.0, -dc / np.maximum(dv * mid, 1e-30)))
        candidate = np.maximum(
            min_density,
            np.maximum(
                density - move,
                np.minimum(1.0, np.minimum(density + move, density * scale)),
            ),
        )
        if float(candidate.mean()) > volume_fraction:
            lo = mid
        else:
            hi = mid
            updated = candidate
        if hi > 0 and (hi - lo) / hi < 1e-4:
            break
    return updated


def run_simp_oc(
    config: FEMConfig,
    *,
    n_iter: int,
    move: float,
    min_density: float,
    tolerance: float,
) -> dict[str, np.ndarray | float | int]:
    """Run SIMP with optimality-criteria updates."""
    evaluator = FEMCantileverEvaluator(config)
    density = np.full(
        (config.grid_height, config.grid_width),
        config.volume_max,
        dtype=np.float64,
    )
    history: list[tuple[int, float, float, float]] = []

    try:
        for iteration in range(1, n_iter + 1):
            _u, compliance, ce = solve_displacement(evaluator, density)
            dc = (
                -config.simp_p
                * (config.e_max - config.e_min)
                * np.maximum(density, min_density) ** (config.simp_p - 1.0)
                * ce
            )
            next_density = oc_update(
                density,
                dc,
                volume_fraction=config.volume_max,
                move=move,
                min_density=min_density,
            )
            change = float(np.max(np.abs(next_density - density)))
            density = next_density
            history.append((iteration, compliance, float(density.mean()), change))
            if change < tolerance:
                break

        _u, final_compliance, _ce = solve_displacement(evaluator, density)
        solid = float(evaluator.solid_compliance)
        return {
            "design": density.astype(np.float32),
            "history": np.asarray(history, dtype=np.float64),
            "compliance": float(final_compliance),
            "solid_compliance": solid,
            "relative_compliance": float(final_compliance / solid),
            "volume": float(density.mean()),
            "iterations": int(history[-1][0]) if history else 0,
        }
    finally:
        evaluator.close()


def save_plot(result: dict[str, np.ndarray | float | int], output_path: Path) -> None:
    design = np.asarray(result["design"])
    history = np.asarray(result["history"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].imshow(design, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0].set_title(
        f"SIMP/OC design\nC={float(result['compliance']):.3f}, "
        f"rel={float(result['relative_compliance']):.3f}, vol={float(result['volume']):.3f}"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    if history.size:
        axes[1].plot(history[:, 0], history[:, 1], label="compliance")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("compliance")
        axes[1].grid(True, alpha=0.25)
    axes[1].set_title("OC convergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SIMP/OC topopt baseline.")
    parser.add_argument("--grid_width", type=int, default=40)
    parser.add_argument("--grid_height", type=int, default=20)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--move", type=float, default=0.2)
    parser.add_argument("--min_density", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--density_filter_radius", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/topopt_simp_oc")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = FEMConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        volume_max=args.volume_max,
        simp_p=args.simp_p,
        e_min=args.e_min,
        encoding="sorted_material",
        sorted_material_profile="binary",
        density_filter_radius=args.density_filter_radius,
        projection_beta=0.0,
    )
    result = run_simp_oc(
        config,
        n_iter=args.n_iter,
        move=args.move,
        min_density=args.min_density,
        tolerance=args.tolerance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "simp_oc_baseline.npz"
    png_path = args.output_dir / "simp_oc_baseline.png"
    np.savez(
        npz_path,
        design=np.asarray(result["design"], dtype=np.float32),
        history=np.asarray(result["history"], dtype=np.float64),
        compliance=np.asarray([result["compliance"]], dtype=np.float64),
        solid_compliance=np.asarray([result["solid_compliance"]], dtype=np.float64),
        relative_compliance=np.asarray(
            [result["relative_compliance"]], dtype=np.float64
        ),
        volume=np.asarray([result["volume"]], dtype=np.float64),
        iterations=np.asarray([result["iterations"]], dtype=np.int32),
        grid_width=np.asarray([args.grid_width], dtype=np.int32),
        grid_height=np.asarray([args.grid_height], dtype=np.int32),
        volume_max=np.asarray([args.volume_max], dtype=np.float64),
        simp_p=np.asarray([args.simp_p], dtype=np.float64),
        e_min=np.asarray([args.e_min], dtype=np.float64),
        move=np.asarray([args.move], dtype=np.float64),
        min_density=np.asarray([args.min_density], dtype=np.float64),
    )
    save_plot(result, png_path)
    print(f"saved_npz={npz_path}")
    print(f"saved_png={png_path}")
    print(f"iterations={int(result['iterations'])}")
    print(f"compliance={float(result['compliance']):.6f}")
    print(f"relative_compliance={float(result['relative_compliance']):.6f}")
    print(f"volume={float(result['volume']):.6f}")


if __name__ == "__main__":
    main()
