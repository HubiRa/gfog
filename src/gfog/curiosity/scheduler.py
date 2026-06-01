import math
from typing import Callable


class Scheduler:
    def __init__(self, schedule_fn: Callable[[int, int], float], total_steps: int):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        self.schedule_fn = schedule_fn
        self.total_steps = total_steps
        self.step_count = 0

    def step(self) -> float:
        clamped_step = min(self.step_count, self.total_steps)
        value = self.schedule_fn(clamped_step, self.total_steps)
        self.step_count += 1
        return value

    def reset(self) -> None:
        self.step_count = 0


def _clamp_unit_interval(x: float) -> float:
    return max(0.0, min(1.0, x))


def warmup_cosine(
    step: int,
    total_steps: int,
    warmup_frac: float = 0.1,
    base: float = 1.0,
    min_val: float = 0.0,
) -> float:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    warmup_steps = max(0, min(int(total_steps * warmup_frac), total_steps))
    step = min(max(step, 0), total_steps)

    if warmup_steps > 0 and step < warmup_steps:
        return base * step / warmup_steps

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = _clamp_unit_interval((step - warmup_steps) / decay_steps)
    return min_val + (base - min_val) * 0.5 * (1 + math.cos(math.pi * progress))


def WarmupCosine(
    total_steps: int,
    warmup_frac: float = 0.1,
    base: float = 1.0,
    min_val: float = 0.0,
) -> Scheduler:
    return Scheduler(
        lambda step, total: warmup_cosine(
            step,
            total,
            warmup_frac=warmup_frac,
            base=base,
            min_val=min_val,
        ),
        total_steps=total_steps,
    )


def warmup_cosine_annealing(
    step: int,
    total: int,
    cycles: int = 4,
    base: float = 1.0,
    min_val: float = 0.0,
    warmup_frac: float | None = None,
    decay: str | float | None = None,
) -> float:
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    if cycles <= 0:
        raise ValueError(f"cycles must be > 0, got {cycles}")

    warmup_steps = int(total * warmup_frac) if warmup_frac else 0
    warmup_steps = max(0, min(warmup_steps, total))
    step = min(max(step, 0), total)

    if warmup_steps > 0 and step < warmup_steps:
        return base * step / warmup_steps

    remaining_steps = total - warmup_steps
    if remaining_steps <= 0:
        return min_val

    cycle_len = max(remaining_steps // cycles, 1)
    after_warmup = min(step - warmup_steps, remaining_steps)
    cycle_idx = min(after_warmup // cycle_len, cycles - 1)
    cycle_step = min(after_warmup % cycle_len, cycle_len)

    if decay is None:
        peak = base
    elif decay == "linear":
        frac = 1.0 - (cycle_idx / max(cycles - 1, 1))
        peak = min_val + (base - min_val) * frac
    elif isinstance(decay, (int, float)):
        peak = base * (float(decay) ** cycle_idx)
    else:
        raise ValueError("decay must be None, 'linear', or a numeric factor")

    progress = _clamp_unit_interval(cycle_step / cycle_len)
    return min_val + (peak - min_val) * 0.5 * (1 + math.cos(math.pi * progress))


def WarmupCosineAnnealing(
    total_steps: int,
    cycles: int = 4,
    warmup_frac: float = 0.1,
    base: float = 1.0,
    min_val: float = 0.0,
    decay: str | float | None = None,
) -> Scheduler:
    return Scheduler(
        lambda step, total: warmup_cosine_annealing(
            step,
            total,
            cycles,
            warmup_frac=warmup_frac,
            base=base,
            min_val=min_val,
            decay=decay,
        ),
        total_steps=total_steps,
    )


def cosine_ramp(
    step: int,
    total_steps: int,
    base: float = 1.0,
    min_val: float = 0.0,
) -> float:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    step = min(max(step, 0), total_steps)
    progress = _clamp_unit_interval(step / total_steps)
    return min_val + (base - min_val) * 0.5 * (1 - math.cos(math.pi * progress))


def CosineRamp(
    total_steps: int,
    base: float = 1.0,
    min_val: float = 0.0,
) -> Scheduler:
    return Scheduler(
        lambda step, total: cosine_ramp(
            step,
            total,
            base=base,
            min_val=min_val,
        ),
        total_steps=total_steps,
    )
