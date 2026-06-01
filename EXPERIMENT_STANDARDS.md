# GFog Experiment Standards

This file records the current default experiment style so new examples do not
silently regress to older GAN/Adam/raw-output settings.

## Portable Defaults

- Prefer ranking objectives for black-box optimization experiments. The current
  baseline is `QuantileRankedDefaultOpt` with `ranker_target_curve=exp`,
  `ranker_tau=4`, `ranker_steps=1`, and `ranker_sample_mode=random_top_pool`.
- Use one discriminator/ranker step by default. Tune the generator/discriminator
  learning-rate ratio instead of adding multiple discriminator steps.
- Prefer Muon/Muon for MLP-heavy `G` and `D` unless a task-specific reason says
  otherwise. The default TTUR-style starting point is `g_lr=0.03`, `d_lr=0.1`.
- Keep `G` and `D` in candidate/latent representation space. Hard decoding,
  sorting, binarization, simulation, or local repair belongs inside `f`.
- Do not use straight-through estimators between `G` and `f` unless an
  experiment is explicitly about that estimator.

## Output Normalization

Distance-based curiosity only makes sense in a controlled representation space.
For new experiments, do not apply raw uniformity or Plummer-style repulsion to
unbounded generator outputs.

Use `gfog.models.OutputNormalizer` around `G` when the black-box decoder is
invariant to a sample-wise shift and positive scaling of the emitted score
vector:

```python
from gfog.models import OutputNormalizer

G = OutputNormalizer(G, "centered_l2")
```

This is the standard for score-vector decoders such as:

- sorted-material topology encodings,
- TSP argsort route scores,
- categorical sequence logits decoded by `argmax`.

For task-specific positive-vector decoders, use a decoder-compatible
normalization instead. Example: the BinHead crypto diagnostic uses
`softplus_l2`, because the black-box decoder expects positive vectors.

Do not force `centered_l2` onto bounded continuous control parameters or
symbolic-regression genomes without a task-specific check; those decoders are
not generally shift/scale invariant.

## Curiosity

- Curiosity should be a small controlled perturbation, not the main objective.
- If curiosity compares against the buffer, normalize the actual generated
  representation first, not only the curiosity loss internals.
- Start with `curiosity_reference=buffer` for elite repulsion when the task has
  stable candidate semantics. Use batch-only curiosity when repelling from
  elites would fight exploitation.

## What Should Be Logged

Every experiment should save or print at least:

- optimizer objective and ranker settings,
- `G`/`D` optimizer names and learning rates,
- batch size and buffer size/multiplier,
- generator output normalization,
- curiosity weight/reference,
- best buffer score and mean/median buffer score over time.
