import argparse
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TorchFEMCantileverConfig:
    grid_width: int = 32
    grid_height: int = 16
    flip_lr_input: bool = False
    simp_p: float = 3.0
    e_min: float = 1e-3
    e_max: float = 1.0
    poisson_ratio: float = 0.3
    volume_max: float = 0.48
    density_filter_radius: int = 1
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    hard_binarize: bool = False
    load_scale: float = 1.0
    youngs_modulus: float = 1.0
    thickness: float = 1.0
    device: str = "cpu"
    dtype: str = "float64"


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


def decode_design_logits_numpy(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


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


class TorchFEMCantileverEvaluator:
    def __init__(self, config: TorchFEMCantileverConfig) -> None:
        ensure_torchfem_importable(None)
        from torchfem import Planar
        from torchfem.materials import IsotropicElasticityPlaneStress
        from torchfem.mesh import rect_quad

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        torch.set_default_dtype(self.dtype)

        self.nelx = config.grid_width
        self.nely = config.grid_height
        self.nelems = self.nelx * self.nely

        nodes, elements = rect_quad(
            self.nelx + 1, self.nely + 1, float(self.nelx), float(self.nely)
        )
        self.nodes = nodes.to(self.device, self.dtype)
        self.elements = elements.to(self.device)
        self.Planar = Planar
        self.material_class = IsotropicElasticityPlaneStress
        self.filter_offsets, self.filter_weights = self._build_filter_kernel()
        self.solid_compliance = self._compute_solid_compliance()

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

    def decode_design(self, raw_logits: np.ndarray) -> np.ndarray:
        sample = decode_design_logits_numpy(raw_logits.reshape(self.nely, self.nelx))
        if self.config.flip_lr_input:
            sample = np.fliplr(sample)
        sample = self._apply_density_filter(sample)
        sample = apply_projection_numpy(
            sample,
            beta=self.config.projection_beta,
            eta=self.config.projection_eta,
            hard_binarize=self.config.hard_binarize,
        )
        return sample

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
        model = self.Planar(
            self.nodes, self.elements, material, thickness=self.config.thickness
        )

        left = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].min())
        right = torch.isclose(model.nodes[:, 0], model.nodes[:, 0].max())
        model.constraints[left, :] = True
        candidates = torch.where(right)[0]
        target_y = torch.tensor(self.nely / 2.0, device=self.device, dtype=self.dtype)
        tip = candidates[torch.argmin(torch.abs(model.nodes[candidates, 1] - target_y))]
        model.forces[tip, 1] = -self.config.load_scale
        return model

    def compliance(self, density_phys: np.ndarray) -> float:
        model = self._build_model(density_phys)
        u, f, _, _, _ = model.solve(method="spsolve")
        return float(torch.inner(f.ravel(), u.ravel()).item())

    def _compute_solid_compliance(self) -> float:
        solid = np.ones((self.nely, self.nelx), dtype=np.float64)
        solid = self._apply_density_filter(solid)
        solid = apply_projection_numpy(
            solid,
            beta=self.config.projection_beta,
            eta=self.config.projection_eta,
            hard_binarize=self.config.hard_binarize,
        )
        return self.compliance(solid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="torch-fem cantilever prototype")
    parser.add_argument("--torchfem_src", type=str, default=None)
    parser.add_argument("--grid_width", type=int, default=32)
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flip_lr_input", action="store_true")
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--e_max", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.3)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument("--projection_beta", type=float, default=0.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument("--load_scale", type=float, default=1.0)
    parser.add_argument("--youngs_modulus", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--input_npz", type=Path, default=None)
    return parser


def load_npz_payload(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if args.input_npz is not None:
        data = np.load(args.input_npz)
        if "raw_design_code" in data:
            raw = np.asarray(data["raw_design_code"], dtype=np.float64)
        elif "raw_design_logits" in data:
            raw = np.asarray(data["raw_design_logits"], dtype=np.float64)
        else:
            raise ValueError(f"No raw design field found in {args.input_npz}")

        actual_compliance = None
        relative_compliance = None
        if "actual_compliance" in data:
            actual_compliance = np.asarray(
                data["actual_compliance"], dtype=np.float64
            ).reshape(-1)
        if "relative_compliance" in data:
            relative_compliance = np.asarray(
                data["relative_compliance"], dtype=np.float64
            ).reshape(-1)
        return (
            raw.reshape(-1, args.grid_height, args.grid_width),
            actual_compliance,
            relative_compliance,
        )

    rng = np.random.default_rng(args.seed)
    logits = rng.standard_normal((1, args.grid_height, args.grid_width))
    flat = logits.reshape(1, -1)
    k = int(round(args.volume_max * flat.shape[1]))
    mask = np.zeros_like(flat)
    top_idx = np.argpartition(flat[0], -k)[-k:]
    mask[0, top_idx] = 1.0
    raw = (mask.reshape(1, args.grid_height, args.grid_width) - 0.5) * 6.0
    return raw, None, None


def rankdata_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)[::-1]
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def main() -> None:
    args = build_parser().parse_args()
    ensure_torchfem_importable(args.torchfem_src)

    cfg = TorchFEMCantileverConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        flip_lr_input=args.flip_lr_input,
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
        dtype=args.dtype,
    )
    evaluator = TorchFEMCantileverEvaluator(cfg)
    raw_logits_batch, ref_actual, ref_relative = load_npz_payload(args)
    tfm_compliance = []
    tfm_relative = []

    print(f"grid={args.grid_width}x{args.grid_height}")
    print(f"solid_compliance={evaluator.solid_compliance:.6f}")
    header = "idx\tmean_density\ttorchfem_comp\ttorchfem_rel"
    if ref_actual is not None:
        header += "\tref_comp\tcomp_delta"
    if ref_relative is not None:
        header += "\tref_rel\trel_delta"
    print(header)

    for idx, raw_logits in enumerate(raw_logits_batch):
        density = evaluator.decode_design(raw_logits)
        compliance = evaluator.compliance(density)
        relative = compliance / evaluator.solid_compliance
        tfm_compliance.append(compliance)
        tfm_relative.append(relative)

        row = f"{idx}\t{density.mean():.6f}\t{compliance:.6f}\t{relative:.6f}"
        if ref_actual is not None:
            row += f"\t{ref_actual[idx]:.6f}\t{(compliance - ref_actual[idx]):.6f}"
        if ref_relative is not None:
            row += f"\t{ref_relative[idx]:.6f}\t{(relative - ref_relative[idx]):.6f}"
        print(row)

    tfm_compliance_np = np.asarray(tfm_compliance, dtype=np.float64)
    tfm_relative_np = np.asarray(tfm_relative, dtype=np.float64)
    if ref_actual is not None and ref_actual.size == tfm_compliance_np.size:
        pearson = float(np.corrcoef(ref_actual, tfm_compliance_np)[0, 1])
        spearman = float(
            np.corrcoef(rankdata_desc(ref_actual), rankdata_desc(tfm_compliance_np))[
                0, 1
            ]
        )
        mae = float(np.mean(np.abs(tfm_compliance_np - ref_actual)))
        print(f"compliance_pearson={pearson:.6f}")
        print(f"compliance_spearman={spearman:.6f}")
        print(f"compliance_mae={mae:.6f}")
    if ref_relative is not None and ref_relative.size == tfm_relative_np.size:
        pearson = float(np.corrcoef(ref_relative, tfm_relative_np)[0, 1])
        spearman = float(
            np.corrcoef(rankdata_desc(ref_relative), rankdata_desc(tfm_relative_np))[
                0, 1
            ]
        )
        mae = float(np.mean(np.abs(tfm_relative_np - ref_relative)))
        print(f"relative_pearson={pearson:.6f}")
        print(f"relative_spearman={spearman:.6f}")
        print(f"relative_mae={mae:.6f}")


if __name__ == "__main__":
    main()
