"""Run GFog on small Nevergrad benchmark functions.

This is intentionally a thin bridge: Nevergrad provides the black-box function,
GFog provides a learned proposal distribution, and optional Nevergrad optimizers
provide baselines under the same number of objective evaluations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nevergrad as ng
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import BaseOpt, components, make_torch_optimizer
from gfog.opt.latents_sampler import LatentSamplerLambda


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dimension: int
    lower: float | None
    upper: float | None
    function: object


class TanhScaledGenerator(nn.Module):
    """Bound an MLP output to the benchmark domain."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        lower: float | None,
        upper: float | None,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim, hidden_dim],
        )
        self.lower = lower
        self.upper = upper

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        if self.lower is None or self.upper is None:
            return x
        midpoint = 0.5 * (self.upper + self.lower)
        radius = 0.5 * (self.upper - self.lower)
        return midpoint + radius * torch.tanh(x)


class NevergradObjective:
    """Vectorized torch-facing wrapper around a Nevergrad ExperimentFunction."""

    def __init__(self, function: object) -> None:
        self.function = function
        self.evaluation_count = 0

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        values = []
        for row in candidates.detach().cpu().numpy():
            values.append(float(self.function(np.asarray(row, dtype=np.float64))))
        self.evaluation_count += len(values)
        return torch.as_tensor(values, dtype=candidates.dtype, device=candidates.device)


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


def rank_by_values(candidates: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.detach().reshape(-1), descending=False)
    return candidates[order]


class RankedOpt(BaseOpt):
    """Rank-target optimizer matching the current GFog experiment style."""

    def __init__(
        self,
        opt_components: components.OptComponents,
        *,
        ranker_list_size: int,
        ranker_sample_pool_size: int,
        ranker_tau: float,
        mixed_rank_update: bool,
    ) -> None:
        super().__init__(opt_components)
        self.ranker_list_size = ranker_list_size
        self.ranker_sample_pool_size = ranker_sample_pool_size
        self.ranker_tau = ranker_tau
        self.mixed_rank_update = mixed_rank_update

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

    def _ranked_buffer_subset_with_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranked = self.buffer.B.get_top_k(min(self.ranker_list_size, len(self.buffer.B)))
        values = torch.as_tensor(
            [row[0] for row in self.buffer.B.get_sorted_values()[: ranked.shape[0]]],
            dtype=self.gan.dtype,
            device=self.gan.device,
        )
        return ranked.to(self.gan.device, self.gan.dtype), values

    def _train_discriminator_on_ranked(self, ranked: torch.Tensor) -> None:
        self.gan.optimizerD.zero_grad()
        real_scores = self.gan.D(ranked)
        targets = rank_targets(
            real_scores.numel(),
            device=real_scores.device,
            dtype=real_scores.dtype,
            tau=self.ranker_tau,
        ).reshape_as(real_scores)
        loss = F.mse_loss(real_scores, targets)
        loss.backward()
        self.gan.optimizerD.step()

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
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()
        return proposals

    def evaluate(self, proposals: torch.Tensor) -> None:
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))

    def step(self) -> None:
        if not self.mixed_rank_update:
            return super().step()

        with torch.no_grad():
            z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
            evaluated = self.gan.G(z)
            values = self.fn.f(evaluated.detach().to(self.fn.device, self.fn.dtype))

        buffer_batch, buffer_values = self._ranked_buffer_subset_with_values()
        mixed = torch.cat([buffer_batch, evaluated.detach()], dim=0)
        mixed_values = torch.cat(
            [buffer_values, values.to(self.gan.device, self.gan.dtype)], dim=0
        )
        ranked = rank_by_values(mixed, mixed_values)
        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_on_ranked(ranked)

        self.gan.optimizerG.zero_grad()
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        proposals = self.gan.G(z)
        scores = self.gan.D(proposals)
        loss = F.mse_loss(scores, torch.ones_like(scores))
        if self.gan.curiosity_loss is not None:
            loss = loss + self.gan.curiosity_loss(proposals)
        loss.backward()
        self.gan.optimizerG.step()

        self.buffer.B.insert_many(values=list(values), tensors=list(evaluated.detach()))


def make_benchmark(args: argparse.Namespace) -> BenchmarkSpec:
    if args.task == "topology":
        from nevergrad.functions import topology_optimization

        function = topology_optimization.TO(args.topology_n)
        return BenchmarkSpec(
            name=f"topology_n{args.topology_n}",
            dimension=int(function.dimension),
            lower=-1.0,
            upper=1.0,
            function=function,
        )
    if args.task == "artificial":
        from nevergrad.functions import ArtificialFunction

        function = ArtificialFunction(
            name=args.function,
            block_dimension=args.dimension,
            rotation=args.rotation,
            bounded=args.bounded,
            noise_level=args.noise,
        )
        return BenchmarkSpec(
            name=f"{args.function}_d{args.dimension}",
            dimension=int(function.dimension),
            lower=-5.0 if args.bounded else None,
            upper=5.0 if args.bounded else None,
            function=function,
        )
    raise ValueError(f"Unknown task: {args.task}")


def build_gfog(
    args: argparse.Namespace, spec: BenchmarkSpec
) -> tuple[RankedOpt, NevergradObjective]:
    device = torch.device("cpu")
    objective = NevergradObjective(spec.function)
    fn = components.Fn(
        f=objective,
        input_dim=spec.dimension,
        device=device,
        dtype=torch.float32,
    )
    g = TanhScaledGenerator(
        input_dim=args.latent_dim,
        output_dim=spec.dimension,
        hidden_dim=args.hidden_dim,
        lower=spec.lower,
        upper=spec.upper,
    ).to(device)
    d = MLP(
        input_dim=spec.dimension,
        output_dim=1,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    ).to(device)
    buffer = components.BufferComp(
        B=Buffer(buffer_size=args.buffer_multiplier * args.batch_size)
    )
    gan = components.GAN(
        G=g,
        D=d,
        loss=BCEWithLogitsLoss(),
        curiosity_loss=None,
        latent_dim=args.latent_dim,
        optimizerG=make_torch_optimizer(
            args.g_optimizer,
            g.parameters(),
            lr=args.g_lr,
            momentum=args.optimizer_momentum,
        ),
        optimizerD=make_torch_optimizer(
            args.d_optimizer,
            d.parameters(),
            lr=args.d_lr,
            momentum=args.optimizer_momentum,
        ),
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
    )
    optimizer = RankedOpt(
        opt_components,
        ranker_list_size=args.ranker_list_size,
        ranker_sample_pool_size=args.ranker_sample_pool_size,
        ranker_tau=args.ranker_tau,
        mixed_rank_update=args.mixed_rank_update,
    )
    return optimizer, objective


def run_nevergrad_baseline(
    *,
    spec: BenchmarkSpec,
    optimizer_name: str,
    budget: int,
    seed: int,
) -> float:
    parametrization = ng.p.Array(shape=(spec.dimension,))
    if spec.lower is not None and spec.upper is not None:
        parametrization = ng.p.Array(
            shape=(spec.dimension,), lower=spec.lower, upper=spec.upper
        )
    parametrization.random_state.seed(seed)
    optimizer_cls = ng.optimizers.registry[optimizer_name]
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget)
    best = float("inf")
    for _ in range(budget):
        candidate = optimizer.ask()
        value = float(spec.function(np.asarray(candidate.value, dtype=np.float64)))
        optimizer.tell(candidate, value)
        best = min(best, value)
    recommended = optimizer.recommend()
    recommended_value = float(
        spec.function(np.asarray(recommended.value, dtype=np.float64))
    )
    return min(best, recommended_value)


def parse_baselines(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    missing = [name for name in names if name not in ng.optimizers.registry]
    if missing:
        raise ValueError(f"Unknown Nevergrad optimizer(s): {missing}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["topology", "artificial"], default="topology"
    )
    parser.add_argument("--topology_n", type=int, default=8)
    parser.add_argument("--function", type=str, default="rastrigin")
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--n_iter", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_multiplier", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.1)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument(
        "--mixed_rank_update",
        action="store_true",
        help="Evaluate G first, then train D on the true-score ranking of buffer samples plus G outputs.",
    )
    parser.add_argument("--baselines", type=str, default="RandomSearch,OnePlusOne,CMA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/nevergrad_benchmarks")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = make_benchmark(args)
    optimizer, objective = build_gfog(args, spec)
    optimizer.optimize(args.n_iter, verbose=True)

    gfog_best = float(optimizer.buffer.B.get_value(0))
    gfog_budget = int(objective.evaluation_count)
    baseline_results = {}
    for name in parse_baselines(args.baselines):
        baseline_results[name] = run_nevergrad_baseline(
            spec=spec,
            optimizer_name=name,
            budget=gfog_budget,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"nevergrad_{spec.name}_seed_{args.seed}.npz"
    np.savez_compressed(
        output_path,
        task=np.asarray([args.task]),
        benchmark=np.asarray([spec.name]),
        dimension=np.asarray([spec.dimension], dtype=np.int32),
        seed=np.asarray([args.seed], dtype=np.int32),
        gfog_best=np.asarray([gfog_best], dtype=np.float32),
        gfog_budget=np.asarray([gfog_budget], dtype=np.int32),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        g_lr=np.asarray([args.g_lr], dtype=np.float32),
        d_lr=np.asarray([args.d_lr], dtype=np.float32),
        mixed_rank_update=np.asarray([args.mixed_rank_update]),
        baseline_names=np.asarray(list(baseline_results.keys())),
        baseline_values=np.asarray(list(baseline_results.values()), dtype=np.float32),
        best_candidate=optimizer.buffer.B.get_top_k(1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy(),
    )
    print(f"benchmark={spec.name}")
    print(f"dimension={spec.dimension}")
    print(f"g_optimizer={args.g_optimizer}")
    print(f"d_optimizer={args.d_optimizer}")
    print(f"g_lr={args.g_lr}")
    print(f"d_lr={args.d_lr}")
    print(f"mixed_rank_update={args.mixed_rank_update}")
    print(f"gfog_budget={gfog_budget}")
    print(f"gfog_best={gfog_best:.8f}")
    for name, value in baseline_results.items():
        print(f"baseline_{name}={value:.8f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
