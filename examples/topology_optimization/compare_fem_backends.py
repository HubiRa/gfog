import argparse

import numpy as np
import torch
import torch.nn.functional as F

from cantilever_fem import FEMCantileverEvaluator, FEMConfig, apply_projection_numpy
from cantilever_torchfem import (
    TorchFEMCantileverConfig,
    TorchFEMCantileverEvaluator,
    ensure_torchfem_importable,
)


def make_smooth_random(nely: int, nelx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse_h = max(2, nely // 4)
    coarse_w = max(2, nelx // 4)
    coarse = rng.random((1, 1, coarse_h, coarse_w), dtype=np.float32)
    up = F.interpolate(
        torch.from_numpy(coarse),
        size=(nely, nelx),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    arr = up.cpu().numpy().astype(np.float64)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-12)
    return arr


def build_test_fields(
    nely: int, nelx: int, volume_max: float, seed: int
) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    fields["solid"] = np.ones((nely, nelx), dtype=np.float64)
    fields["uniform_volfrac"] = np.full((nely, nelx), volume_max, dtype=np.float64)

    smooth = make_smooth_random(nely, nelx, seed)
    fields["random_smooth"] = smooth

    topk = np.zeros((nely, nelx), dtype=np.float64)
    k = int(round(volume_max * topk.size))
    flat = smooth.ravel()
    topk.ravel()[np.argpartition(flat, -k)[-k:]] = 1.0
    fields["random_topk"] = topk

    horiz = np.zeros((nely, nelx), dtype=np.float64)
    band_h = max(1, int(round(volume_max * nely)))
    r0 = max(0, nely // 2 - band_h // 2)
    horiz[r0 : r0 + band_h, :] = 1.0
    fields["horizontal_band"] = horiz

    diag = np.zeros((nely, nelx), dtype=np.float64)
    thickness = max(1, int(round(volume_max * min(nely, nelx) / 2)))
    for r in range(nely):
        c = int(round((nelx - 1) * r / max(nely - 1, 1)))
        c0 = max(0, c - thickness)
        c1 = min(nelx, c + thickness + 1)
        diag[r, c0:c1] = 1.0
    fields["diagonal_band"] = diag

    left_half = np.zeros((nely, nelx), dtype=np.float64)
    left_half[:, : max(1, nelx // 2)] = 1.0
    fields["left_half_solid"] = left_half

    right_half = np.zeros((nely, nelx), dtype=np.float64)
    right_half[:, nelx - max(1, nelx // 2) :] = 1.0
    fields["right_half_solid"] = right_half

    vertical_strip = np.zeros((nely, nelx), dtype=np.float64)
    strip_w = max(1, int(round(volume_max * nelx)))
    vertical_strip[:, :strip_w] = 1.0
    fields["left_vertical_strip"] = vertical_strip

    center_strip = np.zeros((nely, nelx), dtype=np.float64)
    c0 = max(0, nelx // 2 - strip_w // 2)
    center_strip[:, c0 : c0 + strip_w] = 1.0
    fields["center_vertical_strip"] = center_strip

    single_soft = np.full((nely, nelx), volume_max, dtype=np.float64)
    single_soft[nely // 2, nelx // 2] = min(1.0, volume_max + 0.4)
    fields["uniform_plus_center"] = single_soft

    single_void = np.full((nely, nelx), volume_max, dtype=np.float64)
    single_void[nely // 2, nelx // 2] = max(0.0, volume_max - 0.4)
    fields["uniform_minus_center"] = single_void
    return fields


def preprocess_density(
    field: np.ndarray, scipy_eval: FEMCantileverEvaluator
) -> np.ndarray:
    density = scipy_eval._apply_density_filter(field.astype(np.float64, copy=False))
    density = apply_projection_numpy(
        density,
        beta=scipy_eval.config.projection_beta,
        eta=scipy_eval.config.projection_eta,
        hard_binarize=scipy_eval.config.hard_binarize,
    )
    return density


def apply_transform(field: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return field
    if name == "flip_lr":
        return np.fliplr(field)
    if name == "flip_ud":
        return np.flipud(field)
    if name == "flip_both":
        return np.flipud(np.fliplr(field))
    if name == "transpose":
        return field.T
    if name == "transpose_flip_lr":
        return np.fliplr(field.T)
    if name == "transpose_flip_ud":
        return np.flipud(field.T)
    if name == "transpose_flip_both":
        return np.flipud(np.fliplr(field.T))
    raise ValueError(f"Unknown transform: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SciPy FEM and torch-fem backends"
    )
    parser.add_argument("--torchfem_src", type=str, default=None)
    parser.add_argument("--grid_width", type=int, default=40)
    parser.add_argument("--grid_height", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--e_max", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.3)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument("--projection_beta", type=float, default=1.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument("--load_scale", type=float, default=1.0)
    parser.add_argument("--youngs_modulus", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--transform_sweep", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_torchfem_importable(args.torchfem_src)

    scipy_cfg = FEMConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        simp_p=args.simp_p,
        e_min=args.e_min,
        e_max=args.e_max,
        poisson_ratio=args.poisson_ratio,
        volume_max=args.volume_max,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
        load_scale=args.load_scale,
    )
    torch_cfg = TorchFEMCantileverConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        simp_p=args.simp_p,
        e_min=args.e_min,
        e_max=args.e_max,
        poisson_ratio=args.poisson_ratio,
        volume_max=args.volume_max,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
        load_scale=args.load_scale,
        youngs_modulus=args.youngs_modulus,
        thickness=args.thickness,
        device=args.device,
    )
    scipy_eval = FEMCantileverEvaluator(scipy_cfg)
    torch_eval = TorchFEMCantileverEvaluator(torch_cfg)

    fields = build_test_fields(
        args.grid_height, args.grid_width, args.volume_max, args.seed
    )

    print(f"grid={args.grid_width}x{args.grid_height}")
    print(f"scipy_solid_compliance={scipy_eval.solid_compliance:.6f}")
    print(f"torchfem_solid_compliance={torch_eval.solid_compliance:.6f}")
    print(
        "name\tmean_density\tscipy_comp\ttorchfem_comp\tcomp_delta\tscipy_rel\ttorchfem_rel\trel_delta"
    )

    for name, field in fields.items():
        density = preprocess_density(field, scipy_eval)
        scipy_comp = float(scipy_eval._solve_compliance(density))
        torch_comp = float(torch_eval.compliance(density))
        scipy_rel = scipy_comp / scipy_eval.solid_compliance
        torch_rel = torch_comp / torch_eval.solid_compliance
        print(
            f"{name}\t{density.mean():.6f}\t{scipy_comp:.6f}\t{torch_comp:.6f}\t{(torch_comp - scipy_comp):.6f}"
            f"\t{scipy_rel:.6f}\t{torch_rel:.6f}\t{(torch_rel - scipy_rel):.6f}"
        )

    if args.transform_sweep:
        transform_names = [
            "identity",
            "flip_lr",
            "flip_ud",
            "flip_both",
        ]
        print()
        print("transform sweep (torch-fem only on transformed density)")
        print("field\tbest_transform\tscipy_comp\tbest_torchfem_comp\tabs_delta")
        for field_name in [
            "left_half_solid",
            "right_half_solid",
            "left_vertical_strip",
            "center_vertical_strip",
            "horizontal_band",
        ]:
            base = preprocess_density(fields[field_name], scipy_eval)
            scipy_comp = float(scipy_eval._solve_compliance(base))
            best_name = ""
            best_comp = 0.0
            best_delta = float("inf")
            for transform_name in transform_names:
                transformed = apply_transform(base, transform_name)
                torch_comp = float(torch_eval.compliance(transformed))
                delta = abs(torch_comp - scipy_comp)
                if delta < best_delta:
                    best_delta = delta
                    best_name = transform_name
                    best_comp = torch_comp
            print(
                f"{field_name}\t{best_name}\t{scipy_comp:.6f}\t{best_comp:.6f}\t{best_delta:.6f}"
            )


if __name__ == "__main__":
    main()
