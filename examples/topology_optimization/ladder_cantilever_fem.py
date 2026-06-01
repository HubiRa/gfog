import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from cantilever_fem import (
    FEMCantileverEvaluator,
    FEMConfig,
    TorchFEMCantileverEvaluator,
    expand_design_code_numpy,
    make_optimizer,
    make_torch_optimizer,
    pairwise_hamming_mean,
    pairwise_l2_mean,
    save_design_grid,
)
from gfog.buffer import Buffer, Levels
from gfog.curiosity import WangIsolaUniformity, WangIsolaUniformityConfig
from gfog.models import MLP
from gfog.opt import components
from gfog.opt.latents_sampler import LatentSamplerLambda


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolution ladder FEM cantilever experiment"
    )
    parser.add_argument("--backend", choices=["scipy", "torchfem"], default="scipy")
    parser.add_argument("--torchfem_src", type=str, default=None)
    parser.add_argument("--torchfem_device", type=str, default="cpu")

    parser.add_argument("--stage1_grid_width", type=int, default=20)
    parser.add_argument("--stage1_grid_height", type=int, default=10)
    parser.add_argument("--stage1_n_iter", type=int, default=200)
    parser.add_argument("--stage2_grid_width", type=int, default=40)
    parser.add_argument("--stage2_grid_height", type=int, default=20)
    parser.add_argument("--stage2_n_iter", type=int, default=800)
    parser.add_argument("--warmstart_top_k", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument(
        "--generator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument(
        "--discriminator_hidden_dims", nargs="+", type=int, default=[128, 128]
    )
    parser.add_argument("--curiosity", type=float, default=40.0)
    parser.add_argument(
        "--optimizer_type",
        choices=["default", "hinge", "lsgan", "wgan", "wgangp"],
        default="lsgan",
    )
    parser.add_argument(
        "--g_torch_optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="adam",
    )
    parser.add_argument(
        "--d_torch_optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default="sgd",
    )
    parser.add_argument("--g_lr", type=float, default=0.003)
    parser.add_argument("--d_lr", type=float, default=0.01)
    parser.add_argument("--g_momentum", type=float, default=0.9)
    parser.add_argument("--d_momentum", type=float, default=0.9)
    parser.add_argument("--discriminator_steps", type=int, default=3)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--gradient_penalty_weight", type=float, default=10.0)

    parser.add_argument("--volume_max", type=float, default=0.48)
    parser.add_argument("--roughness_max", type=float, default=0.18)
    parser.add_argument("--simp_p", type=float, default=3.0)
    parser.add_argument("--e_min", type=float, default=1e-3)
    parser.add_argument("--e_max", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.3)
    parser.add_argument("--density_filter_radius", type=int, default=1)
    parser.add_argument("--projection_beta", type=float, default=1.0)
    parser.add_argument("--projection_eta", type=float, default=0.5)
    parser.add_argument("--hard_binarize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/fem_cantilever_ladder")
    )
    return parser


def make_evaluator(args: argparse.Namespace, *, grid_width: int, grid_height: int):
    cfg = FEMConfig(
        grid_width=grid_width,
        grid_height=grid_height,
        encoding="direct",
        simp_p=args.simp_p,
        e_min=args.e_min,
        e_max=args.e_max,
        poisson_ratio=args.poisson_ratio,
        volume_max=args.volume_max,
        roughness_max=args.roughness_max,
        density_filter_radius=args.density_filter_radius,
        projection_beta=args.projection_beta,
        projection_eta=args.projection_eta,
        hard_binarize=args.hard_binarize,
    )
    if args.backend == "scipy":
        return cfg, FEMCantileverEvaluator(cfg)
    if args.backend == "torchfem":
        return cfg, TorchFEMCantileverEvaluator(
            cfg,
            torchfem_src=args.torchfem_src,
            device=args.torchfem_device,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def build_optimizer(args: argparse.Namespace, evaluator, f_dim: int):
    device = torch.device("cpu")
    fn = components.Fn(f=evaluator, input_dim=f_dim, device=device, dtype=torch.float32)
    g = MLP(
        input_dim=args.latent_dim,
        output_dim=f_dim,
        hidden_dims=args.generator_hidden_dims,
    ).to(device)
    d = MLP(
        input_dim=f_dim,
        output_dim=1,
        hidden_dims=args.discriminator_hidden_dims,
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
        optimizerG=make_torch_optimizer(
            args.g_torch_optimizer, g.parameters(), args.g_lr, args.g_momentum
        ),
        optimizerD=make_torch_optimizer(
            args.d_torch_optimizer, d.parameters(), args.d_lr, args.d_momentum
        ),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d), b=args.batch_size, d=args.latent_dim
        ),
        device=device,
        dtype=torch.float32,
    )
    optimizer = make_optimizer(
        args.optimizer_type,
        components.OptComponents(
            fn=fn,
            gan=gan,
            batch_size=args.batch_size,
            buffer=buffer,
            discriminator_steps=args.discriminator_steps,
            elite_sampling="random_top_k",
            elite_pool_size=args.buffer_multiplier * args.batch_size,
            weight_clip=args.weight_clip if args.optimizer_type == "wgan" else None,
            gradient_penalty_weight=args.gradient_penalty_weight,
        ),
    )
    return optimizer, buffer, device


def fill_remaining_buffer_random(optimizer, f_dim: int, target_size: int) -> None:
    while len(optimizer.buffer.B) < target_size:
        x = optimizer.gan.latent_sampler().to(optimizer.gan.device, optimizer.gan.dtype)
        with torch.no_grad():
            x = optimizer.gan.G(x)
        values = optimizer.fn.f(x.to(optimizer.fn.device, optimizer.fn.dtype))
        optimizer.buffer.B.insert_many(values=list(values), tensors=list(x.detach()))


def save_stage_outputs(
    stage_name: str, args: argparse.Namespace, evaluator, buffer, output_dir: Path
) -> dict[str, Any]:
    top_k = min(9, len(buffer))
    top_codes = buffer.get_top_k(top_k)
    top_designs = torch.from_numpy(
        evaluator.decode_designs_numpy(top_codes.detach().cpu().numpy())
    ).to(dtype=torch.float32)
    top_values = np.asarray(buffer.get_sorted_values()[:top_k], dtype=np.float32)
    actual_compliance = top_values[:, 2].copy()
    relative_compliance = actual_compliance / evaluator.solid_compliance
    output_dir.mkdir(parents=True, exist_ok=True)
    save_design_grid(
        top_designs,
        actual_compliance,
        relative_compliance,
        top_values,
        output_dir / f"top_designs_seed_{args.seed}.png",
        title=f"{stage_name} top designs (seed={args.seed})",
    )
    np.savez_compressed(
        output_dir / f"top_designs_seed_{args.seed}.npz",
        designs=top_designs.cpu().numpy(),
        raw_design_code=top_codes.cpu().numpy(),
        values=top_values,
        actual_compliance=actual_compliance.astype(np.float32),
        relative_compliance=relative_compliance.astype(np.float32),
        solid_compliance=np.asarray([evaluator.solid_compliance], dtype=np.float32),
    )
    return {
        "top_codes": top_codes.detach().cpu().numpy(),
        "best_compliance": float(actual_compliance[0]),
        "best_relative_compliance": float(relative_compliance[0]),
        "mean_relative_compliance_topk": float(relative_compliance.mean()),
        "mean_l2": pairwise_l2_mean(top_designs),
        "mean_hamming": pairwise_hamming_mean(top_designs),
    }


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Stage 1
    stage1_cfg, stage1_eval = make_evaluator(
        args, grid_width=args.stage1_grid_width, grid_height=args.stage1_grid_height
    )
    stage1_dim = args.stage1_grid_width * args.stage1_grid_height
    stage1_opt, stage1_buffer, _ = build_optimizer(args, stage1_eval, stage1_dim)
    stage1_opt.optimize(args.stage1_n_iter, verbose=True)
    stage1_result = save_stage_outputs(
        "stage1", args, stage1_eval, stage1_buffer.B, args.output_dir / "stage1"
    )

    # Stage 2 with warmstarted buffer
    stage2_cfg, stage2_eval = make_evaluator(
        args, grid_width=args.stage2_grid_width, grid_height=args.stage2_grid_height
    )
    stage2_dim = args.stage2_grid_width * args.stage2_grid_height
    stage2_opt, stage2_buffer, _ = build_optimizer(args, stage2_eval, stage2_dim)
    stage2_buffer.B.clear()

    warm_k = min(
        args.warmstart_top_k,
        stage1_result["top_codes"].shape[0],
        stage2_buffer.B.buffer_size,
    )
    warm_codes_small = np.asarray(stage1_result["top_codes"][:warm_k], dtype=np.float32)
    warm_codes_large = expand_design_code_numpy(
        warm_codes_small,
        coarse_height=args.stage1_grid_height,
        coarse_width=args.stage1_grid_width,
        full_height=args.stage2_grid_height,
        full_width=args.stage2_grid_width,
    ).reshape(warm_k, stage2_dim)
    warm_tensors = [
        torch.from_numpy(code.astype(np.float32)) for code in warm_codes_large
    ]
    warm_values = list(
        stage2_eval(torch.from_numpy(warm_codes_large.astype(np.float32)))
    )
    stage2_buffer.B.insert_many(tensors=warm_tensors, values=warm_values)
    fill_remaining_buffer_random(stage2_opt, stage2_dim, stage2_buffer.B.buffer_size)

    stage2_opt.optimize(args.stage2_n_iter, verbose=True)
    stage2_result = save_stage_outputs(
        "stage2", args, stage2_eval, stage2_buffer.B, args.output_dir / "stage2"
    )

    total_iter = args.stage1_n_iter + args.stage2_n_iter
    print(
        f"Ladder summary: stage1={args.stage1_grid_width}x{args.stage1_grid_height} iter={args.stage1_n_iter}, "
        f"stage2={args.stage2_grid_width}x{args.stage2_grid_height} iter={args.stage2_n_iter}, total_iter={total_iter}"
    )
    print(
        f"Stage1 best_rel={stage1_result['best_relative_compliance']:.4f} | "
        f"Stage2 best_rel={stage2_result['best_relative_compliance']:.4f} | "
        f"Stage2 mean_rel_topk={stage2_result['mean_relative_compliance_topk']:.4f} | "
        f"Stage2 mean_hamming={stage2_result['mean_hamming']:.4f}"
    )


if __name__ == "__main__":
    main()
