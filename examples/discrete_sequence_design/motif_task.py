"""High-dimensional discrete motif optimization with GFog.

This is a cheap, batched, non-differentiable sequence-design benchmark. GFog
proposes categorical sequences, the black-box objective scores hard token
motifs, and baselines use the same evaluation budget.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gfog.buffer import Buffer
from gfog.models import OutputNormalizer


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
    """Small experimental Muon optimizer for matrix-heavy networks."""

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


def make_optimizer(
    optimizer_name: str,
    parameters: Any,
    *,
    lr: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    if optimizer_name == "muon":
        return Muon(parameters, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


@dataclass(frozen=True)
class MotifProblem:
    length: int
    alphabet_size: int
    motif_length: int
    n_motifs: int
    seed: int
    position_mode: str

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")
        if self.alphabet_size <= 1:
            raise ValueError(f"alphabet_size must be > 1, got {self.alphabet_size}")
        if self.motif_length <= 0:
            raise ValueError(f"motif_length must be > 0, got {self.motif_length}")
        if self.n_motifs <= 0:
            raise ValueError(f"n_motifs must be > 0, got {self.n_motifs}")
        if self.n_motifs * self.motif_length > self.length:
            raise ValueError("motifs must fit into the sequence")
        if self.position_mode not in {"fixed", "anywhere"}:
            raise ValueError(
                f"position_mode must be fixed or anywhere, got {self.position_mode}"
            )

    def make_instance(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed)
        positions = (
            torch.linspace(
                0,
                self.length - self.motif_length,
                self.n_motifs,
                dtype=torch.float32,
            )
            .round()
            .to(torch.long)
        )
        motifs = torch.randint(
            self.alphabet_size,
            (self.n_motifs, self.motif_length),
            generator=gen,
        )
        interaction = torch.randint(
            self.alphabet_size,
            (self.n_motifs,),
            generator=gen,
        )
        return positions, motifs, interaction


class MotifObjective:
    """Maximize motif matches and motif interaction tokens."""

    def __init__(self, problem: MotifProblem) -> None:
        self.problem = problem
        self.positions, self.motifs, self.interaction = problem.make_instance()
        self.evaluation_count = 0

    def tokens_from_flat(self, candidates: torch.Tensor) -> torch.Tensor:
        return decode_flat_tokens(
            candidates,
            length=self.problem.length,
            alphabet_size=self.problem.alphabet_size,
        )

    def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(torch.long).cpu()
        scores = torch.zeros(tokens.shape[0], dtype=torch.float32)
        full_matches = torch.ones(
            tokens.shape[0], self.problem.n_motifs, dtype=torch.bool
        )
        if self.problem.position_mode == "anywhere":
            windows = tokens.unfold(1, self.problem.motif_length, 1)
            for motif_idx in range(self.problem.n_motifs):
                matches = windows == self.motifs[motif_idx].reshape(1, 1, -1)
                window_fraction = matches.to(torch.float32).mean(dim=2)
                scores += window_fraction.max(dim=1).values
                full_matches[:, motif_idx] = matches.all(dim=2).any(dim=1)
            if self.problem.n_motifs > 1:
                scores += 2.0 * (full_matches[:, :-1] & full_matches[:, 1:]).to(
                    torch.float32
                ).sum(dim=1)
            composition_hits = tokens == self.interaction[0]
            scores += 0.1 * composition_hits.to(torch.float32).mean(dim=1)
            return scores

        for motif_idx, pos in enumerate(self.positions.tolist()):
            window = tokens[:, pos : pos + self.problem.motif_length]
            matches = window == self.motifs[motif_idx].reshape(1, -1)
            match_fraction = matches.to(torch.float32).mean(dim=1)
            full_matches[:, motif_idx] = matches.all(dim=1)
            scores += match_fraction

        # Add sparse epistasis: adjacent motifs only pay off when both are exact.
        if self.problem.n_motifs > 1:
            scores += 2.0 * (full_matches[:, :-1] & full_matches[:, 1:]).to(
                torch.float32
            ).sum(dim=1)

        # Add weak global composition signal so random search is not purely needle-like.
        composition_hits = tokens == self.interaction[0]
        scores += 0.1 * composition_hits.to(torch.float32).mean(dim=1)
        return scores

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens_from_flat(candidates)
        scores = self.score_tokens(tokens)
        self.evaluation_count += tokens.shape[0]
        return -scores.to(candidates.device, candidates.dtype)


def decode_flat_tokens(
    candidates: torch.Tensor, *, length: int, alphabet_size: int
) -> torch.Tensor:
    decoded = candidates.reshape(candidates.shape[0], length, alphabet_size)
    return torch.argmax(decoded, dim=-1)


class ConvSequenceGenerator(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        length: int,
        alphabet_size: int,
        hidden_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.length = length
        self.alphabet_size = alphabet_size
        self.temperature = temperature
        self.project = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * length),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, alphabet_size, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).reshape(z.shape[0], -1, self.length)
        logits = self.net(x).transpose(1, 2)
        return logits.reshape(z.shape[0], self.length * self.alphabet_size)


class ConvSequenceDiscriminator(nn.Module):
    def __init__(self, *, length: int, alphabet_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.length = length
        self.alphabet_size = alphabet_size
        self.features = nn.Sequential(
            nn.Conv1d(alphabet_size, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.reshape(x.shape[0], self.length, self.alphabet_size).transpose(1, 2)
        return self.score(self.features(seq))


class MLPSequenceGenerator(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        length: int,
        alphabet_size: int,
        hidden_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.length = length
        self.alphabet_size = alphabet_size
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, length * alphabet_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z).reshape(z.shape[0], self.length, self.alphabet_size)
        return logits.reshape(z.shape[0], self.length * self.alphabet_size)


class MLPSequenceDiscriminator(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEditSequenceGenerator(nn.Module):
    """Elite-conditioned edit generator over raw sequence logits."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        length: int,
        alphabet_size: int,
        hidden_dim: int,
        edit_scale: float,
    ) -> None:
        super().__init__()
        self.length = length
        self.alphabet_size = alphabet_size
        self.edit_scale = edit_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, elite: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([elite, z], dim=1))
        return elite + self.edit_scale * torch.tanh(delta)


class MLPTokenEditSequenceGenerator(nn.Module):
    """Elite-conditioned token editor with differentiable soft edit output."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        length: int,
        alphabet_size: int,
        hidden_dim: int,
        mutation_temperature: float,
        token_temperature: float,
        mutation_bias: float,
    ) -> None:
        super().__init__()
        self.length = length
        self.alphabet_size = alphabet_size
        self.mutation_temperature = mutation_temperature
        self.token_temperature = token_temperature
        self.mutation_bias = mutation_bias
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, length * (alphabet_size + 1)),
        )

    def forward(self, elite: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        elite_tokens = decode_flat_tokens(
            elite,
            length=self.length,
            alphabet_size=self.alphabet_size,
        )
        elite_one_hot = F.one_hot(elite_tokens, num_classes=self.alphabet_size).to(
            elite.dtype
        )
        raw = self.net(torch.cat([elite_one_hot.reshape(elite.shape[0], -1), z], dim=1))
        raw = raw.reshape(elite.shape[0], self.length, self.alphabet_size + 1)
        mutation_logits = raw[..., 0] + self.mutation_bias
        replacement_logits = raw[..., 1:]
        mutation_prob = torch.sigmoid(
            mutation_logits / self.mutation_temperature
        ).unsqueeze(-1)
        replacement_probs = torch.softmax(
            replacement_logits / self.token_temperature, dim=-1
        )
        edited = (
            1.0 - mutation_prob
        ) * elite_one_hot + mutation_prob * replacement_probs
        return edited.reshape(elite.shape[0], self.length * self.alphabet_size)


class TransformerTokenEditSequenceGenerator(nn.Module):
    """Transformer controller that outputs sparse token-edit actions."""

    def __init__(
        self,
        *,
        latent_dim: int,
        length: int,
        alphabet_size: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mutation_temperature: float,
        token_temperature: float,
        mutation_bias: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by heads; got {hidden_dim=} {heads=}"
            )
        self.length = length
        self.alphabet_size = alphabet_size
        self.mutation_temperature = mutation_temperature
        self.token_temperature = token_temperature
        self.mutation_bias = mutation_bias
        self.token_embedding = nn.Linear(alphabet_size, hidden_dim)
        self.noise_projection = nn.Linear(latent_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.edit_head = nn.Linear(hidden_dim, 1)
        self.replacement_head = nn.Linear(hidden_dim, alphabet_size)

    def forward(self, elite: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        elite_tokens = decode_flat_tokens(
            elite,
            length=self.length,
            alphabet_size=self.alphabet_size,
        )
        elite_one_hot = F.one_hot(elite_tokens, num_classes=self.alphabet_size).to(
            elite.dtype
        )
        noise_context = self.noise_projection(z).unsqueeze(1)
        x = (
            self.token_embedding(elite_one_hot)
            + self.position_embedding
            + noise_context
        )
        encoded = self.encoder(x)
        mutation_logits = self.edit_head(encoded).squeeze(-1) + self.mutation_bias
        replacement_logits = self.replacement_head(encoded)
        mutation_prob = torch.sigmoid(
            mutation_logits / self.mutation_temperature
        ).unsqueeze(-1)
        replacement_probs = torch.softmax(
            replacement_logits / self.token_temperature, dim=-1
        )
        edited = (
            1.0 - mutation_prob
        ) * elite_one_hot + mutation_prob * replacement_probs
        return edited.reshape(elite.shape[0], self.length * self.alphabet_size)


def rank_targets(
    n: int, *, device: torch.device, dtype: torch.dtype, tau: float
) -> torch.Tensor:
    if n <= 1:
        return torch.ones(n, device=device, dtype=dtype)
    positions = torch.arange(n, device=device, dtype=dtype)
    targets = torch.exp(-positions / tau)
    return targets / targets[0].clamp_min(1e-8)


def rank_by_values(
    candidates: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(values.detach().reshape(-1), descending=False)
    return candidates[order], values[order]


class RankedSequenceGFog:
    def __init__(
        self,
        *,
        objective: MotifObjective,
        generator: nn.Module,
        discriminator: nn.Module,
        batch_size: int,
        buffer_size: int,
        latent_dim: int,
        ranker_list_size: int,
        ranker_sample_pool_size: int,
        ranker_tau: float,
        g_lr: float,
        d_lr: float,
        g_optimizer: str,
        d_optimizer: str,
        optimizer_momentum: float,
        edit_mode: bool,
        use_fake_loss: bool,
        mixed_rank_update: bool,
        device: torch.device,
    ) -> None:
        self.objective = objective
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.ranker_list_size = ranker_list_size
        self.ranker_sample_pool_size = ranker_sample_pool_size
        self.ranker_tau = ranker_tau
        self.edit_mode = edit_mode
        self.use_fake_loss = use_fake_loss
        self.mixed_rank_update = mixed_rank_update
        self.device = device
        self.optimizerG = make_optimizer(
            g_optimizer,
            self.G.parameters(),
            lr=g_lr,
            momentum=optimizer_momentum,
        )
        self.optimizerD = make_optimizer(
            d_optimizer,
            self.D.parameters(),
            lr=d_lr,
            momentum=optimizer_momentum,
        )
        self.buffer = Buffer(buffer_size=buffer_size)
        self.best_scores: list[float] = []
        self.eval_counts: list[int] = []
        self._init_buffer()

    def _sample_latent(self) -> torch.Tensor:
        return torch.randn(self.batch_size, self.latent_dim, device=self.device)

    def _sample_random_logits(self, n: int | None = None) -> torch.Tensor:
        batch = self.batch_size if n is None else n
        input_dim = self.objective.problem.length * self.objective.problem.alphabet_size
        return torch.randn(batch, input_dim, device=self.device)

    def _sample_elites(self) -> torch.Tensor:
        k = min(len(self.buffer), max(self.batch_size, self.ranker_sample_pool_size))
        positions = torch.randint(k, (self.batch_size,)).tolist()
        elites = torch.stack([self.buffer.get(int(pos)) for pos in positions])
        return elites.to(self.device)

    def _generate(self) -> torch.Tensor:
        z = self._sample_latent()
        if self.edit_mode:
            return self.G(self._sample_elites(), z)
        return self.G(z)

    def _init_buffer(self) -> None:
        n_batches = int(np.ceil(self.buffer.buffer_size / self.batch_size))
        for _ in range(n_batches):
            with torch.no_grad():
                candidates = self._sample_random_logits(self.batch_size)
            values = self.objective(candidates)
            self.buffer.insert_many(
                tensors=list(candidates.detach()), values=list(values)
            )

    def _ranked_buffer_subset(self) -> torch.Tensor:
        current_len = len(self.buffer)
        pool_size = min(
            current_len, max(self.ranker_list_size, self.ranker_sample_pool_size)
        )
        k = min(self.ranker_list_size, pool_size)
        if k == pool_size:
            ranked = self.buffer.get_top_k(k)
        else:
            positions = torch.randperm(pool_size)[:k].sort().values.tolist()
            ranked = torch.stack([self.buffer.get(int(pos)) for pos in positions])
        return ranked.to(self.device)

    def _ranked_buffer_subset_with_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranked = self.buffer.get_top_k(min(self.ranker_list_size, len(self.buffer))).to(
            self.device
        )
        values = torch.as_tensor(
            [row[0] for row in self.buffer.get_sorted_values()[: ranked.shape[0]]],
            dtype=torch.float32,
            device=self.device,
        )
        return ranked, values

    def step(self) -> None:
        evaluated_proposals = None
        evaluated_values = None
        if self.mixed_rank_update:
            with torch.no_grad():
                evaluated_proposals = self._generate()
                evaluated_values = self.objective(evaluated_proposals.detach())

        self.optimizerD.zero_grad()
        if self.mixed_rank_update:
            assert evaluated_proposals is not None
            assert evaluated_values is not None
            buffer_batch, buffer_values = self._ranked_buffer_subset_with_values()
            mixed = torch.cat([buffer_batch, evaluated_proposals.detach()], dim=0)
            mixed_values = torch.cat(
                [
                    buffer_values.to(evaluated_values.dtype),
                    evaluated_values.to(self.device),
                ],
                dim=0,
            )
            ranked, _ = rank_by_values(mixed, mixed_values)
        else:
            ranked = self._ranked_buffer_subset()
        real_scores = self.D(ranked)
        targets = rank_targets(
            real_scores.numel(),
            device=real_scores.device,
            dtype=real_scores.dtype,
            tau=self.ranker_tau,
        ).reshape_as(real_scores)
        real_loss = F.mse_loss(real_scores, targets)
        d_loss = real_loss
        if self.use_fake_loss:
            with torch.no_grad():
                fake = self._generate()
            fake_scores = self.D(fake.detach())
            d_loss = d_loss + F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
        d_loss.backward()
        self.optimizerD.step()

        self.optimizerG.zero_grad()
        proposals = self._generate()
        g_loss = F.mse_loss(
            self.D(proposals), torch.ones(self.batch_size, 1, device=self.device)
        )
        g_loss.backward()
        self.optimizerG.step()

        with torch.no_grad():
            if self.mixed_rank_update:
                assert evaluated_proposals is not None
                assert evaluated_values is not None
                values = evaluated_values
                proposals_to_insert = evaluated_proposals
            else:
                values = self.objective(proposals.detach())
                proposals_to_insert = proposals
        self.buffer.insert_many(
            tensors=list(proposals_to_insert.detach()), values=list(values)
        )
        self.best_scores.append(-float(self.buffer.get_value(0)))
        self.eval_counts.append(self.objective.evaluation_count)


def random_search(
    objective: MotifObjective, *, budget: int, batch_size: int
) -> tuple[float, list[float]]:
    best = -float("inf")
    curve = []
    remaining = budget
    while remaining > 0:
        n = min(batch_size, remaining)
        tokens = torch.randint(
            objective.problem.alphabet_size,
            (n, objective.problem.length),
        )
        scores = objective.score_tokens(tokens)
        best = max(best, float(scores.max().item()))
        curve.append(best)
        remaining -= n
    return best, curve


def elite_mutation(
    objective: MotifObjective,
    *,
    budget: int,
    batch_size: int,
    elite_size: int,
    mutation_rate: float,
) -> tuple[float, list[float]]:
    init_n = min(batch_size, budget)
    tokens = torch.randint(
        objective.problem.alphabet_size,
        (init_n, objective.problem.length),
    )
    scores = objective.score_tokens(tokens)
    remaining = budget - init_n
    curve = [float(scores.max().item())]
    while remaining > 0:
        n = min(batch_size, remaining)
        k = min(elite_size, tokens.shape[0])
        elite_idx = torch.topk(scores, k=k).indices
        parents = tokens[elite_idx[torch.randint(k, (n,))]].clone()
        mask = torch.rand_like(parents.to(torch.float32)) < mutation_rate
        replacements = torch.randint(
            objective.problem.alphabet_size,
            parents.shape,
        )
        children = torch.where(mask, replacements, parents)
        child_scores = objective.score_tokens(children)
        tokens = torch.cat([tokens, children], dim=0)
        scores = torch.cat([scores, child_scores], dim=0)
        top = torch.topk(scores, k=min(max(elite_size * 4, batch_size), scores.numel()))
        tokens = tokens[top.indices]
        scores = top.values
        curve.append(float(scores.max().item()))
        remaining -= n
    return float(scores.max().item()), curve


def cem(
    objective: MotifObjective,
    *,
    budget: int,
    batch_size: int,
    elite_fraction: float,
    smoothing: float,
) -> tuple[float, list[float]]:
    probs = torch.full(
        (objective.problem.length, objective.problem.alphabet_size),
        1.0 / objective.problem.alphabet_size,
    )
    best = -float("inf")
    curve = []
    remaining = budget
    while remaining > 0:
        n = min(batch_size, remaining)
        tokens = torch.multinomial(
            probs, num_samples=n, replacement=True
        ).T.contiguous()
        scores = objective.score_tokens(tokens)
        best = max(best, float(scores.max().item()))
        k = max(1, int(elite_fraction * n))
        elites = tokens[torch.topk(scores, k=k).indices]
        counts = torch.zeros_like(probs)
        counts.scatter_add_(1, elites.T, torch.ones(objective.problem.length, k))
        new_probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = (1.0 - smoothing) * probs + smoothing * new_probs
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        curve.append(best)
        remaining -= n
    return best, curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--alphabet_size", type=int, default=8)
    parser.add_argument("--motif_length", type=int, default=8)
    parser.add_argument("--n_motifs", type=int, default=8)
    parser.add_argument(
        "--position_mode", choices=["fixed", "anywhere"], default="fixed"
    )
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--buffer_multiplier", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--generator_output_norm",
        choices=["none", "l2", "centered_l2", "layernorm"],
        default="centered_l2",
        help="Normalize generated logits before D/f. centered_l2 preserves argmax decoding.",
    )
    parser.add_argument(
        "--model_type",
        choices=["conv", "mlp", "edit_mlp", "token_edit_mlp", "transformer_token_edit"],
        default="conv",
    )
    parser.add_argument("--edit_scale", type=float, default=2.0)
    parser.add_argument("--mutation_temperature", type=float, default=1.0)
    parser.add_argument("--token_temperature", type=float, default=1.0)
    parser.add_argument("--mutation_bias", type=float, default=-4.0)
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ranker_list_size", type=int, default=256)
    parser.add_argument("--ranker_sample_pool_size", type=int, default=1024)
    parser.add_argument("--ranker_tau", type=float, default=8.0)
    parser.add_argument("--g_lr", type=float, default=0.03)
    parser.add_argument("--d_lr", type=float, default=0.03)
    parser.add_argument(
        "--g_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument(
        "--d_optimizer", choices=["adam", "adamw", "sgd", "muon"], default="muon"
    )
    parser.add_argument("--optimizer_momentum", type=float, default=0.95)
    parser.add_argument(
        "--no_fake_loss",
        action="store_true",
        help="Train D only on ranked evaluated buffer samples; do not push G outputs to zero.",
    )
    parser.add_argument(
        "--mixed_rank_update",
        action="store_true",
        help="Evaluate G first, then train D on the true-score ranking of buffer samples plus G outputs.",
    )
    parser.add_argument("--mutation_rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/discrete_sequence_design")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    problem = MotifProblem(
        length=args.length,
        alphabet_size=args.alphabet_size,
        motif_length=args.motif_length,
        n_motifs=args.n_motifs,
        seed=args.seed,
        position_mode=args.position_mode,
    )
    objective = MotifObjective(problem)
    input_dim = args.length * args.alphabet_size
    edit_mode = args.model_type in {
        "edit_mlp",
        "token_edit_mlp",
        "transformer_token_edit",
    }
    if args.model_type == "conv":
        generator: nn.Module = ConvSequenceGenerator(
            latent_dim=args.latent_dim,
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
        )
        discriminator: nn.Module = ConvSequenceDiscriminator(
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
        )
    elif args.model_type == "mlp":
        generator = MLPSequenceGenerator(
            latent_dim=args.latent_dim,
            length=args.length,
            alphabet_size=args.alphabet_size,
            hidden_dim=args.hidden_dim,
            temperature=args.temperature,
        )
        discriminator = MLPSequenceDiscriminator(
            input_dim=input_dim, hidden_dim=args.hidden_dim
        )
    else:
        if args.model_type == "edit_mlp":
            generator = MLPEditSequenceGenerator(
                input_dim=input_dim,
                latent_dim=args.latent_dim,
                length=args.length,
                alphabet_size=args.alphabet_size,
                hidden_dim=args.hidden_dim,
                edit_scale=args.edit_scale,
            )
        elif args.model_type == "token_edit_mlp":
            generator = MLPTokenEditSequenceGenerator(
                input_dim=input_dim,
                latent_dim=args.latent_dim,
                length=args.length,
                alphabet_size=args.alphabet_size,
                hidden_dim=args.hidden_dim,
                mutation_temperature=args.mutation_temperature,
                token_temperature=args.token_temperature,
                mutation_bias=args.mutation_bias,
            )
        else:
            generator = TransformerTokenEditSequenceGenerator(
                latent_dim=args.latent_dim,
                length=args.length,
                alphabet_size=args.alphabet_size,
                hidden_dim=args.hidden_dim,
                depth=args.transformer_depth,
                heads=args.transformer_heads,
                mutation_temperature=args.mutation_temperature,
                token_temperature=args.token_temperature,
                mutation_bias=args.mutation_bias,
            )
        discriminator = MLPSequenceDiscriminator(
            input_dim=input_dim, hidden_dim=args.hidden_dim
        )

    generator = OutputNormalizer(generator, args.generator_output_norm)

    gfog = RankedSequenceGFog(
        objective=objective,
        generator=generator,
        discriminator=discriminator,
        batch_size=args.batch_size,
        buffer_size=args.buffer_multiplier * args.batch_size,
        latent_dim=args.latent_dim,
        ranker_list_size=args.ranker_list_size,
        ranker_sample_pool_size=args.ranker_sample_pool_size,
        ranker_tau=args.ranker_tau,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        g_optimizer=args.g_optimizer,
        d_optimizer=args.d_optimizer,
        optimizer_momentum=args.optimizer_momentum,
        edit_mode=edit_mode,
        use_fake_loss=not args.no_fake_loss,
        mixed_rank_update=args.mixed_rank_update,
        device=torch.device("cpu"),
    )
    start = time.perf_counter()
    for _ in range(args.n_iter):
        gfog.step()
    elapsed = time.perf_counter() - start
    gfog_best = -float(gfog.buffer.get_value(0))
    budget = objective.evaluation_count

    baseline_objective = MotifObjective(problem)
    random_best, random_curve = random_search(
        baseline_objective,
        budget=budget,
        batch_size=args.batch_size,
    )
    mutation_best, mutation_curve = elite_mutation(
        baseline_objective,
        budget=budget,
        batch_size=args.batch_size,
        elite_size=min(args.batch_size, 1024),
        mutation_rate=args.mutation_rate,
    )
    cem_best, cem_curve = cem(
        baseline_objective,
        budget=budget,
        batch_size=args.batch_size,
        elite_fraction=0.1,
        smoothing=0.5,
    )

    best_logits = gfog.buffer.get_top_k(1)
    best_tokens = decode_flat_tokens(
        best_logits,
        length=args.length,
        alphabet_size=args.alphabet_size,
    ).squeeze(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir
        / f"motif_l{args.length}_a{args.alphabet_size}_seed_{args.seed}.npz"
    )
    np.savez_compressed(
        output_path,
        best_tokens=best_tokens.detach().cpu().numpy(),
        gfog_best=np.asarray([gfog_best], dtype=np.float32),
        random_best=np.asarray([random_best], dtype=np.float32),
        mutation_best=np.asarray([mutation_best], dtype=np.float32),
        cem_best=np.asarray([cem_best], dtype=np.float32),
        gfog_curve=np.asarray(gfog.best_scores, dtype=np.float32),
        gfog_eval_counts=np.asarray(gfog.eval_counts, dtype=np.int64),
        random_curve=np.asarray(random_curve, dtype=np.float32),
        mutation_curve=np.asarray(mutation_curve, dtype=np.float32),
        cem_curve=np.asarray(cem_curve, dtype=np.float32),
        motifs=objective.motifs.numpy(),
        motif_positions=objective.positions.numpy(),
        position_mode=np.asarray([args.position_mode]),
        budget=np.asarray([budget], dtype=np.int64),
        elapsed_seconds=np.asarray([elapsed], dtype=np.float32),
        model_type=np.asarray([args.model_type]),
        generator_output_norm=np.asarray([args.generator_output_norm]),
        edit_scale=np.asarray([args.edit_scale], dtype=np.float32),
        mutation_temperature=np.asarray([args.mutation_temperature], dtype=np.float32),
        token_temperature=np.asarray([args.token_temperature], dtype=np.float32),
        mutation_bias=np.asarray([args.mutation_bias], dtype=np.float32),
        transformer_depth=np.asarray([args.transformer_depth], dtype=np.int32),
        transformer_heads=np.asarray([args.transformer_heads], dtype=np.int32),
        g_optimizer=np.asarray([args.g_optimizer]),
        d_optimizer=np.asarray([args.d_optimizer]),
        use_fake_loss=np.asarray([not args.no_fake_loss]),
        mixed_rank_update=np.asarray([args.mixed_rank_update]),
        config=np.asarray([vars(args)], dtype=object),
    )
    print(f"length={args.length}")
    print(f"alphabet_size={args.alphabet_size}")
    print(f"input_dim={input_dim}")
    print(f"model_type={args.model_type}")
    print(f"position_mode={args.position_mode}")
    print(f"g_optimizer={args.g_optimizer}")
    print(f"d_optimizer={args.d_optimizer}")
    print(f"use_fake_loss={not args.no_fake_loss}")
    print(f"mixed_rank_update={args.mixed_rank_update}")
    print(f"budget={budget}")
    print(f"gfog_best={gfog_best:.6f}")
    print(f"random_best={random_best:.6f}")
    print(f"mutation_best={mutation_best:.6f}")
    print(f"cem_best={cem_best:.6f}")
    print(f"gfog_elapsed_seconds={elapsed:.2f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
