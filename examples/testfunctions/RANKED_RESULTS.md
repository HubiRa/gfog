# Ranked Test Function Checks

Small 2D test functions are useful for checking whether the ranked GFog setup is
basically sane before testing harder black-box problems.

Runner:

```bash
python examples/testfunctions/ranked_testfunctions.py
```

Compared modes:

```text
old ranked:
  D(buffer ranked samples) -> exp(-rank/tau)
  D(G samples) -> 0

mixed rank:
  evaluate G samples first
  rank top buffer samples + evaluated G samples by true objective
  D(mixed ranked samples) -> exp(-rank/tau)
```

Common settings:

```text
n_iter = 500
batch_size = 64
buffer_multiplier = 2
buffer_size = 128
ranker_list_size = 64
ranker_sample_pool_size = 128
ranker_tau = 4
latent_dim = 16
hidden_dim = 64
seed = 0
budget = 32128 evaluations
```

Seed-0 results:

```text
Himmelblau
  old ranked best = 0.000108
  mixed rank best = 0.000875
  random best = 0.000097 / 0.016423 depending run seed stream

Mishra Bird, unconstrained
  old ranked best = -106.763588
  mixed rank best = -106.762238
  random best = -106.756020 / -106.476822 depending run seed stream

Ackley
  old ranked best = 0.076233
  mixed rank best = 0.016525
  random best = 0.080391 / 0.067928 depending run seed stream
```

Interpretation:

```text
Both ranked variants are sane on easy 2D functions.
Mixed rank is not universally better.
It helped Ackley, but old fake-loss ranked was slightly better on Himmelblau and Mishra.
```

This supports the current view from topology: mixed rank is conceptually cleaner
and sometimes better, but the old `D(G)->0` term can act as useful regularization
depending on problem/batch regime.
