"""Optimize vanilla RNN weights on a Shakespeare-style character task.

GFog emits flattened RNN parameters. The objective evaluates those parameters on
fixed character windows and returns next-character cross entropy. The objective
is treated as black-box by GFog, but unlike delayed XOR it is a real text task.
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


FALLBACK_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!
"""


@dataclass(frozen=True)
class CharRNNShape:
    vocab_size: int
    hidden_dim: int
    rnn_param: str = "full"

    @property
    def n_params(self) -> int:
        if self.rnn_param == "readout":
            return self.hidden_dim * self.vocab_size + self.vocab_size
        total = self.vocab_size * self.hidden_dim
        if self.rnn_param == "full":
            total += self.hidden_dim * self.hidden_dim
        elif self.rnn_param != "reservoir":
            raise ValueError(f"Unknown rnn_param: {self.rnn_param}")
        return (
            total
            + self.hidden_dim
            + self.hidden_dim * self.vocab_size
            + self.vocab_size
        )


def load_text(path: Path | None, max_chars: int) -> str:
    text = FALLBACK_SHAKESPEARE if path is None else path.read_text(encoding="utf-8")
    text = text[:max_chars]
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = text.strip()
    if len(text) < 128:
        raise ValueError("text must contain at least 128 characters")
    return text


def make_char_windows(
    *,
    text: str,
    seq_len: int,
    n_sequences: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    chars = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(chars)}
    ids = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    if ids.numel() <= seq_len + 1:
        raise ValueError(f"text is too short for seq_len={seq_len}")
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq_len - 1, (n_sequences,), generator=gen)
    inputs = torch.stack([ids[start : start + seq_len] for start in starts])
    targets = torch.stack([ids[start + 1 : start + seq_len + 1] for start in starts])
    return inputs, targets, chars


class ShakespeareRNNObjective:
    """Black-box objective over flattened char-RNN parameters."""

    def __init__(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        shape: CharRNNShape,
        weight_scale: float,
        score_mode: str,
        seed: int,
        reservoir_radius: float,
        l2_weight: float,
    ) -> None:
        self.inputs = inputs.to(torch.long)
        self.targets = targets.to(torch.long)
        self.shape = shape
        self.weight_scale = weight_scale
        if score_mode not in {"ce", "accuracy"}:
            raise ValueError(
                f"score_mode must be one of ce, accuracy; got {score_mode}"
            )
        self.score_mode = score_mode
        if reservoir_radius <= 0:
            raise ValueError(
                f"reservoir_radius must be positive, got {reservoir_radius}"
            )
        fixed_gen = torch.Generator().manual_seed(seed + 918_771)
        fixed_w_ih = torch.randn(
            shape.vocab_size, shape.hidden_dim, generator=fixed_gen
        )
        self.fixed_w_ih = 0.5 * fixed_w_ih.to(torch.float32) / (shape.hidden_dim**0.5)
        q, _r = torch.linalg.qr(
            torch.randn(shape.hidden_dim, shape.hidden_dim, generator=fixed_gen)
        )
        self.fixed_w_hh = q.to(torch.float32) * reservoir_radius
        self.fixed_b_h = torch.zeros(shape.hidden_dim)
        self.l2_weight = l2_weight

    def _decode_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        weights = self.weight_scale * torch.tanh(params)
        offset = 0
        if self.shape.rnn_param == "readout":
            w_ih = self.fixed_w_ih.expand(params.shape[0], -1, -1)
            w_hh = self.fixed_w_hh.expand(params.shape[0], -1, -1)
            b_h = self.fixed_b_h.expand(params.shape[0], -1)
        else:
            w_ih_size = self.shape.vocab_size * self.shape.hidden_dim
            w_ih = weights[:, offset : offset + w_ih_size].reshape(
                params.shape[0],
                self.shape.vocab_size,
                self.shape.hidden_dim,
            )
            offset += w_ih_size
        if self.shape.rnn_param == "full":
            w_hh_size = self.shape.hidden_dim * self.shape.hidden_dim
            w_hh = weights[:, offset : offset + w_hh_size].reshape(
                params.shape[0],
                self.shape.hidden_dim,
                self.shape.hidden_dim,
            )
            offset += w_hh_size
        elif self.shape.rnn_param == "reservoir":
            w_hh = self.fixed_w_hh.expand(params.shape[0], -1, -1)
        if self.shape.rnn_param != "readout":
            b_h = weights[:, offset : offset + self.shape.hidden_dim]
            offset += self.shape.hidden_dim
        w_out_size = self.shape.hidden_dim * self.shape.vocab_size
        w_out = weights[:, offset : offset + w_out_size].reshape(
            params.shape[0],
            self.shape.hidden_dim,
            self.shape.vocab_size,
        )
        offset += w_out_size
        b_out = weights[:, offset : offset + self.shape.vocab_size]
        return weights, w_ih, w_hh, b_h, w_out, b_out

    def _forward_batch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, w_ih, w_hh, b_h, w_out, b_out = self._decode_batch(params)
        n_candidates = params.shape[0]
        n_sequences = self.inputs.shape[0]
        hidden = torch.zeros(n_candidates, n_sequences, self.shape.hidden_dim)
        logits_by_step = []
        for step in range(self.inputs.shape[1]):
            token_ids = self.inputs[:, step]
            input_term = w_ih[:, token_ids, :]
            recurrent_term = torch.einsum("csh,chd->csd", hidden, w_hh)
            hidden = torch.tanh(input_term + recurrent_term + b_h[:, None, :])
            logits = torch.einsum("csh,chv->csv", hidden, w_out) + b_out[:, None, :]
            logits_by_step.append(logits)
        return torch.stack(logits_by_step, dim=2), weights

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        params = candidates.detach().cpu()
        logits, weights = self._forward_batch(params)
        targets = self.targets[None, :, :]
        if self.score_mode == "ce":
            loss = F.cross_entropy(
                logits.reshape(-1, self.shape.vocab_size),
                targets.expand(logits.shape[0], -1, -1).reshape(-1),
                reduction="none",
            ).reshape(logits.shape[0], -1)
            values = loss.mean(dim=1)
        else:
            pred = torch.argmax(logits, dim=-1)
            values = (pred != targets).to(torch.float32).mean(dim=(1, 2))
        l2 = self.l2_weight * torch.mean(weights * weights, dim=1)
        return (values + l2).to(candidates.device, candidates.dtype)

    def metrics(self, params: torch.Tensor) -> tuple[float, float]:
        logits, _weights = self._forward_batch(params.detach().cpu().reshape(1, -1))
        ce = F.cross_entropy(
            logits.reshape(-1, self.shape.vocab_size), self.targets.reshape(-1)
        )
        pred = torch.argmax(logits[0], dim=-1)
        accuracy = (pred == self.targets).to(torch.float32).mean()
        return float(ce.item()), float(accuracy.item())


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


class RankedOpt(BaseOpt):
    """Rank-target GFog optimizer for scalar black-box objectives."""

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
    ShakespeareRNNObjective,
    list[str],
]:
    device = torch.device("cpu")
    text = load_text(args.text_path, args.max_chars)
    inputs, targets, chars = make_char_windows(
        text=text,
        seq_len=args.seq_len,
        n_sequences=args.n_sequences,
        seed=args.seed,
    )
    shape = CharRNNShape(
        vocab_size=len(chars),
        hidden_dim=args.rnn_hidden_dim,
        rnn_param=args.rnn_param,
    )
    objective = ShakespeareRNNObjective(
        inputs=inputs,
        targets=targets,
        shape=shape,
        weight_scale=args.task_weight_scale,
        score_mode=args.score_mode,
        seed=args.seed,
        reservoir_radius=args.reservoir_radius,
        l2_weight=args.l2_weight,
    )
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
            chars,
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
    return opt_cls(opt_components), objective, chars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=Path, default=None)
    parser.add_argument("--max_chars", type=int, default=20_000)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--n_sequences", type=int, default=256)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_multiplier", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--generator_hidden_dim", type=int, default=256)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=256)
    parser.add_argument("--rnn_hidden_dim", type=int, default=16)
    parser.add_argument(
        "--rnn_param",
        choices=["full", "reservoir", "readout"],
        default="full",
        help="Choose full RNN, fixed reservoir, or fixed reservoir with readout-only genome.",
    )
    parser.add_argument("--reservoir_radius", type=float, default=1.0)
    parser.add_argument("--task_weight_scale", type=float, default=0.5)
    parser.add_argument("--score_mode", choices=["ce", "accuracy"], default="ce")
    parser.add_argument("--l2_weight", type=float, default=1e-5)
    parser.add_argument(
        "--optimizer",
        choices=["default", "lsgan", "wgan", "ranked"],
        default="ranked",
    )
    parser.add_argument("--ranker_list_size", type=int, default=64)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=128)
    parser.add_argument("--ranker_tau", type=float, default=4.0)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=3e-3)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--weight_clip", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/rnn_shakespeare"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer, objective, chars = build_optimizer(args)
    optimizer.optimize(args.n_iter, verbose=True)

    best = optimizer.buffer.B.get_top_k(1).squeeze(0)
    best_value = float(optimizer.buffer.B.get_value(0))
    best_ce, best_accuracy = objective.metrics(best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / f"rnn_shakespeare_seed_{args.seed}.npz",
        best_params=best.detach().cpu().numpy(),
        best_value=np.asarray([best_value], dtype=np.float32),
        best_ce=np.asarray([best_ce], dtype=np.float32),
        best_accuracy=np.asarray([best_accuracy], dtype=np.float32),
        sorted_values=np.asarray(
            optimizer.buffer.B.get_sorted_values(), dtype=np.float32
        ),
        vocab_size=np.asarray([len(chars)], dtype=np.int32),
        rnn_hidden_dim=np.asarray([args.rnn_hidden_dim], dtype=np.int32),
        rnn_param=np.asarray([args.rnn_param]),
        reservoir_radius=np.asarray([args.reservoir_radius], dtype=np.float32),
        seq_len=np.asarray([args.seq_len], dtype=np.int32),
        n_sequences=np.asarray([args.n_sequences], dtype=np.int32),
        score_mode=np.asarray([args.score_mode]),
    )
    print(f"vocab_size={len(chars)}")
    print(f"n_params={best.numel()}")
    print(f"best_value={best_value:.6f}")
    print(f"best_ce={best_ce:.6f}")
    print(f"best_accuracy={best_accuracy:.4f}")
    print(f"saved={args.output_dir / f'rnn_shakespeare_seed_{args.seed}.npz'}")


if __name__ == "__main__":
    main()
