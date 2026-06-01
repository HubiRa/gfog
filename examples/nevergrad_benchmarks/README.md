# Nevergrad Benchmarks

This example runs GFog on benchmark functions provided by
[Nevergrad](https://github.com/facebookresearch/nevergrad), then compares
against selected Nevergrad optimizers under the same objective-evaluation
budget.

GFog uses the current ranked setup:

```text
G(z) -> candidate vector
f(candidate) -> black-box score
D learns rank targets from the buffer
G learns to maximize D(G(z))
```

For bounded Nevergrad tasks, `G` output is squashed with `tanh` into the task
domain.

## Commands

Topology optimization:

```bash
TMPDIR=/private/tmp python examples/nevergrad_benchmarks/nevergrad_gfog.py \
  --task topology \
  --topology_n 8 \
  --n_iter 500 \
  --batch_size 64 \
  --buffer_multiplier 4 \
  --baselines RandomSearch,OnePlusOne,CMA \
  --output_dir results/nevergrad_topology_n8_ranked_500
```

Classic continuous sanity check:

```bash
TMPDIR=/private/tmp python examples/nevergrad_benchmarks/nevergrad_gfog.py \
  --task artificial \
  --function rastrigin \
  --dimension 16 \
  --bounded \
  --n_iter 500 \
  --batch_size 64 \
  --buffer_multiplier 4 \
  --baselines RandomSearch,OnePlusOne,CMA \
  --output_dir results/nevergrad_rastrigin16_ranked_500
```

## Initial Results

Seed 0, ranked MLP-G/MLP-D, 500 GFog iterations, batch size 64:

```text
topology_n8
  dimension = 64
  budget = 32256 evaluations
  GFog = 4.095300
  RandomSearch = 4.809830
  OnePlusOne = 5.000000
  CMA = 5.000000

rastrigin_d16 bounded
  dimension = 16
  budget = 32256 evaluations
  GFog = 61.629509
  RandomSearch = 95.793695
  OnePlusOne = 83.128264
  CMA = 22.884038
```

This is a useful split. GFog looks competitive on the structured topology task,
but loses to CMA on a classic continuous function. That supports the current
hypothesis: GFog should be evaluated on structured expensive black-box design
tasks, not primarily on smooth continuous optimization where specialized
optimizers are strong.

Larger batch topology run:

```bash
TMPDIR=/private/tmp python examples/nevergrad_benchmarks/nevergrad_gfog.py \
  --task topology \
  --topology_n 8 \
  --n_iter 500 \
  --batch_size 512 \
  --buffer_multiplier 4 \
  --ranker_list_size 256 \
  --ranker_sample_pool_size 1024 \
  --hidden_dim 256 \
  --latent_dim 64 \
  --baselines RandomSearch,OnePlusOne,CMA \
  --output_dir results/nevergrad_topology_n8_ranked_b512_500
```

```text
topology_n8, large batch
  dimension = 64
  budget = 258048 evaluations
  GFog = 3.704281
  RandomSearch = 4.744077
  OnePlusOne = 5.000000
  CMA = 5.000000
```

The large batch helped substantially on topology: `4.095300 -> 3.704281`.
This is consistent with the core GFog assumption that learning a proposal
distribution benefits from evaluating many candidates per update.

## Mixed Rank Update

The ranked optimizer also supports:

```bash
--mixed_rank_update
```

In this mode, generated candidates are evaluated before the `D` update. `D` is
then trained on the true-score ranking of:

```text
top buffer samples + evaluated G samples
```

This removes the old implicit fake label `D(G(z)) -> 0`; generated samples are
only ranked low if their true objective value is bad.

Seed-0 topology results:

```text
topology_n8, batch 64, 500 iters
  budget = 32256 evaluations
  old ranked = 4.095300
  mixed rank = 4.004312
  RandomSearch = 4.809830
  OnePlusOne = 5.000000
  CMA = 5.000000

topology_n8, batch 512, 500 iters
  budget = 258048 evaluations
  old ranked = 3.704281
  mixed rank = 4.016545
  RandomSearch = 4.744077
  OnePlusOne = 5.000000
  CMA = 5.000000
```

Mixed ranking is conceptually cleaner and helped at batch 64, but it hurt the
large-batch run. A likely explanation is that at large batch the evaluated G
samples dominate the mixed rank list and weaken the elite-buffer signal. If we
keep this variant, the next knob is the mixture ratio: train `D` on a fixed
number of top buffer samples plus only the best or a stratified subset of G
outputs, rather than the full generated batch.
