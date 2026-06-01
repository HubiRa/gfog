import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer, Levels
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import DefaultOpt, components
from gfog.opt.latents_sampler import LatentSamplerLambda


BackendName = Literal["numpy", "jax"]
BackendDevice = Literal["auto", "cpu", "gpu"]


@dataclass
class StructuralConfig:
    grid_width: int = 32
    grid_height: int = 16
    simp_p: float = 3.0
    e_min: float = 1e-3
    e_max: float = 1.0
    volume_max: float = 0.48
    roughness_max: float = 0.18
    load_scale: float = 1.0
    diagonal_regularization: float = 1e-6
    diagonal_edge_factor: float = 0.7
    density_filter_radius: int = 1
    projection_beta: float = 0.0
    projection_eta: float = 0.5
    hard_binarize: bool = False
    force_axis: int = 1


class ArrayBackend:
    """Small backend wrapper to keep Torch/GFog separate from the solver backend.

    The evaluator communicates with GFog only through ordinary arrays. Today we
    support a NumPy implementation and an optional JAX implementation with the
    same API. This keeps the structural solver fully swappable.
    """

    def __init__(
        self,
        name: BackendName = "numpy",
        device: BackendDevice = "auto",
        jit: bool = True,
    ) -> None:
        self.name = name
        self.device = device
        self.jit = jit

        self.jax = None
        self.jnp = None
        self._jax_device = None
        if name == "jax":
            try:
                import jax
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    "JAX backend requested but jax is not installed. "
                    "Install jax/jaxlib or use --backend numpy."
                ) from exc
            self.jax = jax
            self.jnp = jnp
            self._jax_device = self._pick_jax_device(device)

    def _pick_jax_device(self, device: BackendDevice):
        assert self.jax is not None
        if device == "auto":
            return self.jax.devices()[0]
        if device == "cpu":
            return self.jax.devices("cpu")[0]
        gpu_devices = self.jax.devices("gpu")
        if not gpu_devices:
            raise ValueError(
                "JAX GPU backend requested, but no GPU device is available"
            )
        return gpu_devices[0]

    def asarray(self, x: np.ndarray):
        if self.name == "numpy":
            return np.asarray(x, dtype=np.float32)
        assert self.jnp is not None and self.jax is not None
        arr = self.jnp.asarray(x, dtype=self.jnp.float32)
        return self.jax.device_put(arr, self._jax_device)

    def to_numpy(self, x) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(x)
        assert self.jax is not None
        return np.asarray(self.jax.device_get(x))


def decode_design_logits_numpy(x: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x32))


def apply_projection_numpy(
    x: np.ndarray, beta: float, eta: float, hard_binarize: bool
) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    if beta > 0:
        num = np.tanh(beta * eta) + np.tanh(beta * (out - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        out = num / max(den, 1e-8)
    if hard_binarize:
        out = (out >= eta).astype(np.float32)
    return out


class StructuralCantileverEvaluator:
    """Compliance-style cantilever evaluator with interchangeable NumPy/JAX backends.

    This is intentionally structured so that:
    - GFog remains pure Torch.
    - The structural objective remains a separate black-box evaluator.
    - We can switch between NumPy CPU execution and optional JAX CPU/GPU
      execution without changing GFog internals.

    The generator emits unconstrained logits. The black-box evaluator maps them
    to physical densities with a sigmoid, then applies filtering and solves the
    structural response. This keeps discriminator/GAN gradients away from an
    output squashing nonlinearity while preserving a standard density-based
    design representation inside the mechanics model.
    """

    def __init__(
        self,
        config: StructuralConfig,
        *,
        backend: BackendName = "numpy",
        backend_device: BackendDevice = "auto",
        jit: bool = True,
    ) -> None:
        self.config = config
        self.backend = ArrayBackend(backend, backend_device, jit)
        self.backend_name = backend
        self._node_count = config.grid_width * config.grid_height
        self._dof_count = 2 * self._node_count
        fixed_nodes = np.array(
            [row * config.grid_width for row in range(config.grid_height)],
            dtype=np.int32,
        )
        self._fixed_dofs = np.sort(
            np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        ).astype(np.int32)
        self._free_dofs = np.array(
            [
                i
                for i in range(self._dof_count)
                if i not in set(self._fixed_dofs.tolist())
            ],
            dtype=np.int32,
        )
        self._force = np.zeros(self._dof_count, dtype=np.float32)
        load_node = (config.grid_height // 2) * config.grid_width + (
            config.grid_width - 1
        )
        self._force[2 * load_node + config.force_axis] = -config.load_scale
        self._edges = self._build_edges()
        self._filter_offsets, self._filter_weights = self._build_filter_kernel()

        if backend == "jax":
            self._build_jax_fns()

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
        return offsets, np.asarray(weights, dtype=np.float32)

    def _apply_density_filter_numpy(self, sample: np.ndarray) -> np.ndarray:
        radius = self.config.density_filter_radius
        if radius <= 0:
            return sample
        filtered = np.zeros_like(sample)
        weight_sum = np.zeros_like(sample)
        for (dr, dc), weight in zip(
            self._filter_offsets, self._filter_weights, strict=False
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
        return filtered / np.maximum(weight_sum, 1e-8)

    def _build_edges(self) -> list[tuple[int, int, float, float, float]]:
        edges: list[tuple[int, int, float, float, float]] = []
        h = self.config.grid_height
        w = self.config.grid_width

        for row in range(h):
            for col in range(w - 1):
                a = row * w + col
                b = a + 1
                edges.append((a, b, 1.0, 0.0, 1.0))

        for row in range(h - 1):
            for col in range(w):
                a = row * w + col
                b = a + w
                edges.append((a, b, 0.0, 1.0, 1.0))

        diag = 1.0 / np.sqrt(2.0)
        for row in range(h - 1):
            for col in range(w - 1):
                a = row * w + col
                b = a + w + 1
                edges.append((a, b, diag, diag, self.config.diagonal_edge_factor))

                a = row * w + (col + 1)
                b = a + w - 1
                edges.append((a, b, -diag, diag, self.config.diagonal_edge_factor))

        return edges

    def _build_jax_fns(self) -> None:
        assert self.backend.jax is not None and self.backend.jnp is not None
        jax = self.backend.jax
        jnp = self.backend.jnp
        cfg = self.config
        free = jnp.asarray(self._free_dofs)
        force = jnp.asarray(self._force)
        h = cfg.grid_height
        w = cfg.grid_width
        edges = self._edges
        filter_offsets = self._filter_offsets
        filter_weights = jnp.asarray(self._filter_weights)

        def apply_density_filter(x):
            if cfg.density_filter_radius <= 0:
                return x
            filtered = jnp.zeros_like(x)
            weight_sum = jnp.zeros_like(x)
            for (dr, dc), weight in zip(filter_offsets, filter_weights, strict=False):
                src_r0 = max(0, -dr)
                src_r1 = h - max(0, dr)
                src_c0 = max(0, -dc)
                src_c1 = w - max(0, dc)
                dst_r0 = max(0, dr)
                dst_r1 = dst_r0 + (src_r1 - src_r0)
                dst_c0 = max(0, dc)
                dst_c1 = dst_c0 + (src_c1 - src_c0)
                filtered = filtered.at[dst_r0:dst_r1, dst_c0:dst_c1].add(
                    weight * x[src_r0:src_r1, src_c0:src_c1]
                )
                weight_sum = weight_sum.at[dst_r0:dst_r1, dst_c0:dst_c1].add(weight)
            return filtered / jnp.maximum(weight_sum, 1e-8)

        def one_sample(x_flat):
            x_logits = x_flat.reshape(h, w)
            x_raw = 1.0 / (1.0 + jnp.exp(-x_logits))
            x = apply_density_filter(x_raw)
            if cfg.projection_beta > 0:
                num = jnp.tanh(cfg.projection_beta * cfg.projection_eta) + jnp.tanh(
                    cfg.projection_beta * (x - cfg.projection_eta)
                )
                den = jnp.tanh(cfg.projection_beta * cfg.projection_eta) + jnp.tanh(
                    cfg.projection_beta * (1.0 - cfg.projection_eta)
                )
                x = num / jnp.maximum(den, 1e-8)
            if cfg.hard_binarize:
                x = (x >= cfg.projection_eta).astype(jnp.float32)
            volume = jnp.mean(x)
            dx = jnp.mean(jnp.abs(x[:, 1:] - x[:, :-1]))
            dy = jnp.mean(jnp.abs(x[1:, :] - x[:-1, :]))
            roughness = 0.5 * (dx + dy)

            A = jnp.eye(2 * h * w, dtype=jnp.float32) * cfg.diagonal_regularization

            def add_edge(A_mat, a: int, b: int, k_edge, nx, ny):
                block = k_edge * jnp.array(
                    [
                        [nx * nx, nx * ny, -nx * nx, -nx * ny],
                        [ny * nx, ny * ny, -ny * nx, -ny * ny],
                        [-nx * nx, -nx * ny, nx * nx, nx * ny],
                        [-ny * nx, -ny * ny, ny * nx, ny * ny],
                    ],
                    dtype=jnp.float32,
                )
                dofs = jnp.array([2 * a, 2 * a + 1, 2 * b, 2 * b + 1])
                return A_mat.at[jnp.ix_(dofs, dofs)].add(block)

            for a, b, nx, ny, scale in edges:
                ax, ay = divmod(a, w)
                bx, by = divmod(b, w)
                rho = 0.5 * (x[ax, ay] + x[bx, by])
                k = scale * (cfg.e_min + (rho**cfg.simp_p) * (cfg.e_max - cfg.e_min))
                A = add_edge(A, a, b, k, nx, ny)

            A_ff = A[jnp.ix_(free, free)]
            f_f = force[free]
            u_f = jnp.linalg.solve(A_ff, f_f)
            compliance = jnp.dot(f_f, u_f)

            return jnp.stack(
                [
                    jnp.maximum(volume - cfg.volume_max, 0.0),
                    jnp.maximum(roughness - cfg.roughness_max, 0.0),
                    compliance,
                ]
            )

        batched = jax.vmap(one_sample)
        self._jax_eval = jax.jit(batched) if self.backend.jit else batched

    def decode_designs_numpy(self, x_np: np.ndarray) -> np.ndarray:
        cfg = self.config
        h = cfg.grid_height
        w = cfg.grid_width
        x_phys = decode_design_logits_numpy(x_np)
        decoded = []
        for sample_raw in x_phys.reshape(-1, h, w):
            sample = self._apply_density_filter_numpy(sample_raw)
            sample = apply_projection_numpy(
                sample,
                beta=cfg.projection_beta,
                eta=cfg.projection_eta,
                hard_binarize=cfg.hard_binarize,
            )
            decoded.append(sample)
        return np.asarray(decoded, dtype=np.float32)

    def _numpy_eval(self, x_np: np.ndarray) -> np.ndarray:
        cfg = self.config
        h = cfg.grid_height
        w = cfg.grid_width
        results: list[list[float]] = []

        for sample in self.decode_designs_numpy(x_np).reshape(-1, h, w):
            volume = float(sample.mean())
            dx = float(np.abs(sample[:, 1:] - sample[:, :-1]).mean())
            dy = float(np.abs(sample[1:, :] - sample[:-1, :]).mean())
            roughness = 0.5 * (dx + dy)

            A = np.eye(self._dof_count, dtype=np.float32) * cfg.diagonal_regularization

            for a, b, nx, ny, scale in self._edges:
                ax, ay = divmod(a, w)
                bx, by = divmod(b, w)
                rho = 0.5 * (sample[ax, ay] + sample[bx, by])
                k = scale * (cfg.e_min + (rho**cfg.simp_p) * (cfg.e_max - cfg.e_min))
                block = k * np.array(
                    [
                        [nx * nx, nx * ny, -nx * nx, -nx * ny],
                        [ny * nx, ny * ny, -ny * nx, -ny * ny],
                        [-nx * nx, -nx * ny, nx * nx, nx * ny],
                        [-ny * nx, -ny * ny, ny * nx, ny * ny],
                    ],
                    dtype=np.float32,
                )
                dofs = np.array([2 * a, 2 * a + 1, 2 * b, 2 * b + 1])
                A[np.ix_(dofs, dofs)] += block

            A_ff = A[np.ix_(self._free_dofs, self._free_dofs)]
            f_f = self._force[self._free_dofs]
            u_f = np.linalg.solve(A_ff, f_f)
            compliance = float(f_f @ u_f)

            results.append(
                [
                    max(volume - cfg.volume_max, 0.0),
                    max(roughness - cfg.roughness_max, 0.0),
                    compliance,
                ]
            )
        return np.asarray(results, dtype=np.float32)

    def __call__(self, theta: torch.Tensor | np.ndarray) -> list[list[float]]:
        if isinstance(theta, torch.Tensor):
            x_np = theta.detach().to("cpu", torch.float32).numpy()
        else:
            x_np = np.asarray(theta, dtype=np.float32)

        if self.backend_name == "numpy":
            values = self._numpy_eval(x_np)
        else:
            x_backend = self.backend.asarray(x_np)
            values = self.backend.to_numpy(self._jax_eval(x_backend))
        return values.tolist()


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
    values: list[list[float]],
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
        vol_violation, roughness_violation, objective = values[idx]
        ax.set_title(
            f"#{idx + 1}\nvol={vol_violation:.3f} rough={roughness_violation:.3f} comp={objective:.3f}",
            fontsize=9,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GFog structural cantilever benchmark")
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    parser.add_argument(
        "--backend_device", choices=["auto", "cpu", "gpu"], default="auto"
    )
    parser.add_argument(
        "--jit", action="store_true", help="Use JAX jit when backend=jax"
    )
    parser.add_argument("--grid_width", type=int, default=32)
    parser.add_argument("--grid_height", type=int, default=16)
    parser.add_argument("--n_iter", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--curiosity", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--g_lr", type=float, default=0.01)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument("--discriminator_steps", type=int, default=3)
    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--roughness_max", type=float, default=0.18)
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--diagonal_edge_factor", type=float, default=0.7)
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument("--projection_beta", type=float, default=0.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/structural_cantilever"),
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = StructuralConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        simp_p=args.simp_p,
        volume_max=args.volume_max,
        roughness_max=args.roughness_max,
        diagonal_edge_factor=args.diagonal_edge_factor,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
    )
    evaluator = StructuralCantileverEvaluator(
        cfg,
        backend=args.backend,
        backend_device=args.backend_device,
        jit=args.jit,
    )

    f_dim = args.grid_width * args.grid_height
    device = torch.device("cpu")

    fn = components.Fn(f=evaluator, input_dim=f_dim, device=device, dtype=torch.float32)
    g = MLP(
        input_dim=args.latent_dim,
        output_dim=f_dim,
        hidden_dims=[128, 128],
    ).to(device)
    d = MLP(
        input_dim=f_dim,
        output_dim=1,
        hidden_dims=[128, 128],
        use_spectral_norm=True,
    ).to(device)

    buffer = components.BufferComp(
        B=Buffer(
            buffer_size=args.buffer_multiplier * args.batch_size,
            value_levels=Levels(
                ["volume_violation", "roughness_violation", "compliance"]
            ),
        )
    )

    curiosity_loss = None
    if args.curiosity > 0:
        curiosity_loss = WangIsolaUniformity(
            WangIsolaUniformityConfig(use_buffer=True, weight=args.curiosity),
            buffer=buffer.B,
        )

    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=curiosity_loss,
        latent_dim=args.latent_dim,
        optimizerG=torch.optim.Adam(g.parameters(), lr=args.g_lr),
        optimizerD=torch.optim.Adam(d.parameters(), lr=args.d_lr),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d), b=args.batch_size, d=args.latent_dim
        ),
        device=device,
        dtype=torch.float32,
    )

    optimizer = DefaultOpt(
        components.OptComponents(
            fn=fn,
            gan=gan,
            batch_size=args.batch_size,
            buffer=buffer,
            discriminator_steps=args.discriminator_steps,
            elite_sampling="random_top_k",
            elite_pool_size=args.buffer_multiplier * args.batch_size,
        )
    )

    logger.info(
        f"StructuralCantilever: backend={args.backend} backend_device={args.backend_device} "
        f"grid={args.grid_width}x{args.grid_height} n_iter={args.n_iter} curiosity={args.curiosity} "
        f"projection_beta={args.projection_beta} hard_binarize={args.hard_binarize}"
    )
    logger.info("Initial top-5 buffer values")
    buffer.B.print_values(slice(0, 5, 1))

    optimizer.optimize(args.n_iter, verbose=True)

    logger.info("Final top-5 buffer values")
    buffer.B.print_values(slice(0, 5, 1))

    top_k = min(9, len(buffer.B))
    top_logits = buffer.B.get_top_k(top_k).reshape(
        top_k, args.grid_height, args.grid_width
    )
    top_designs = torch.from_numpy(
        evaluator.decode_designs_numpy(top_logits.detach().cpu().numpy())
    ).to(device=device, dtype=torch.float32)
    top_values = buffer.B.get_sorted_values()[:top_k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"backend_{args.backend}_curiosity_{args.curiosity:g}_seed_{args.seed}"
    save_design_grid(
        top_designs,
        top_values,
        args.output_dir / f"top_designs_{suffix}.png",
        title=(
            f"Structural Cantilever Top Designs "
            f"({args.backend}, curiosity={args.curiosity:g}, seed={args.seed})"
        ),
    )
    np.savez_compressed(
        args.output_dir / f"top_designs_{suffix}.npz",
        designs=top_designs.cpu().numpy(),
        raw_design_logits=top_logits.cpu().numpy(),
        values=np.asarray(top_values, dtype=np.float32),
        curiosity=np.asarray([args.curiosity], dtype=np.float32),
        seed=np.asarray([args.seed], dtype=np.int32),
        backend=np.asarray([args.backend]),
        grid_width=np.asarray([args.grid_width], dtype=np.int32),
        grid_height=np.asarray([args.grid_height], dtype=np.int32),
        buffer_multiplier=np.asarray([args.buffer_multiplier], dtype=np.int32),
        density_filter_radius=np.asarray([args.density_filter_radius], dtype=np.int32),
        projection_beta=np.asarray([args.projection_beta], dtype=np.float32),
        projection_eta=np.asarray([args.projection_eta], dtype=np.float32),
        hard_binarize=np.asarray([args.hard_binarize], dtype=np.int32),
    )

    return {
        "best_value": top_values[0],
        "best_compliance": float(top_values[0][2]),
        "mean_compliance_topk": float(np.mean([row[2] for row in top_values])),
        "mean_l2": pairwise_l2_mean(top_designs),
        "mean_hamming": pairwise_hamming_mean(top_designs),
        "curiosity": args.curiosity,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "artifact_path": str(args.output_dir / f"top_designs_{suffix}.npz"),
        "backend": args.backend,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(args)
    logger.info(f"Saved outputs to {result['output_dir']}")
    logger.info(f"Saved raw artifacts to {result['artifact_path']}")
    logger.info(f"Best buffer value vector: {result['best_value']}")
    logger.info(
        f"Top-k summary: best_compliance={result['best_compliance']:.4f} "
        f"mean_l2={result['mean_l2']:.4f} mean_hamming={result['mean_hamming']:.4f}"
    )


if __name__ == "__main__":
    main()
