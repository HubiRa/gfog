import math


class Scheduler:
    def __init__(self, schedule_fn, total_steps: int):
        self.schedule_fn = schedule_fn
        self.total_steps = total_steps
        self.step_count = 0

    def step(self):
        value = self.schedule_fn(self.step_count, self.total_steps)
        self.step_count += 1
        return value

    def reset(self):
        self.step_count = 0


# ~~~~~~~~~~~~~~~~~~~~~~~~
# warmup with cosine decay
# ~~~~~~~~~~~~~~~~~~~~~~~~


def warmup_cosine(step, total_steps, warmup_frac=0.1, base=1.0, min_val=0.0):
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return base * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_val + (base - min_val) * 0.5 * (1 + math.cos(math.pi * progress))


def WarmupCosine(
    total_steps: int, warmup_frac: float = 0.1, base: float = 1.0, min_val: float = 0.0
) -> Scheduler:
    return Scheduler(
        lambda step, total: warmup_cosine(
            step, total, warmup_frac=warmup_frac, base=base, min_val=min_val
        ),
        total_steps=total_steps,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~
# warmup and cosine annealing (like SGDR)
# ~~~~~~~~~~~~~~~~~~~~~~~~


def warmup_cosine_annealing(
    step, total, cycles=4, base=1.0, min_val=0.0, warmup_frac=None, decay=None
):
    warmup_steps = int(total * warmup_frac) if warmup_frac else 0

    # Warmup phase
    if step < warmup_steps:
        return base * step / warmup_steps if warmup_steps > 0 else base

    # Remaining steps after warmup
    after_warmup = step - warmup_steps
    cycle_len = (total - warmup_steps) // cycles
    cycle_idx = after_warmup // cycle_len
    cycle_step = after_warmup % cycle_len

    # Peak adjustment with decay
    if decay is None:
        peak = base
    elif decay == "linear":
        frac = 1 - cycle_idx / cycles
        peak = min_val + (base - min_val) * frac
    else:  # exponential
        peak = base * (decay**cycle_idx)

    # Cosine annealing
    return min_val + (peak - min_val) * 0.5 * (
        1 + math.cos(math.pi * cycle_step / cycle_len)
    )


def WarmupCosineAnnealing(
    total_steps: int,
    cycles: int = 4,
    warmup_frac: float = 0.1,
    base: float = 1.0,
    min_val: float = 0.0,
    decay: str | None = None,
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
