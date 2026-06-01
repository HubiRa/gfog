# TSP Argsort GFog Experiment Learnings

This note tracks reusable findings for `examples/tsp/tsp_argsort.py`. Lower tour length is better.

## Problem Setup

- Objective `f`: `G(z)` emits one scalar score per city. The black-box objective sorts scores with `argsort` to form a route, then evaluates Euclidean tour length.
- Current playground: `n_cities=50`, `city_seed=0`, `seed=0`, `batch_size=128`, `buffer_multiplier=4`, `ranker_list_size=128`, `ranker_sample_pool_size=512`.
- Baselines for the default city set:
  - Random search best over matched budget: about `20.414`.
  - Nearest neighbor: `7.215`.
  - 2-opt from nearest neighbor: `6.294`.
  - 2-opt from random starts: about `5.791`.

## Current Best Default Result

Best 1000-iteration default-city run so far:

```text
objective: quantile
G/D: MLP 256,256 / 256,256
generator output norm: layernorm
optimizer: Muon / Muon
g_lr: 0.001
d_lr: 0.03
curiosity: 0.03
ranker_tau: 16
batch_size: 64
buffer_multiplier: 4
ranker_list_size: 64
ranker_sample_pool_size: 256
best_length: 8.2901
final_buffer_mean: 8.5886
artifact: results/tsp_argsort_large_sweep/tsp_city0_n50_quantile_gh256x256_dh256x256_b64_bm4_rank64_pool256_tau16_normlayernorm_goptmuon_doptmuon_glr0.001_dlr0.03_curio0.03_batch_ct2_iter1000_seed0.npz
```

Previous best before the large LR/capacity/tau/norm sweep:

```text
objective: quantile
G/D: MLP 128,128 / 128,128
generator output norm: layernorm
optimizer: Muon / Muon
g_lr: 0.001
d_lr: 0.01
curiosity: 0.03
ranker_tau: 8
best_length: 10.9502
```

Previous best after the wide LR-only sweep:

```text
objective: quantile
G/D: MLP 128,128 / 128,128
generator output norm: layernorm
optimizer: Muon / Muon
g_lr: 0.001
d_lr: 0.03
curiosity: 0.3
ranker_tau: 8
best_length: 10.8661
```

## Main Findings

- TTUR matters a lot. Equal or near-equal `G`/`D` learning rates are bad on this task. `D` needs to move materially faster than `G`.
- The first LR grid was too conservative. Larger `D` LR and some larger `G` LR settings are viable.
- Best LR pair in the large sweep was again `g_lr=0.001`, `d_lr=0.03`; `g_lr=0.003`, `d_lr=0.1` was the next strongest high-LR region.
- High LR settings can work:
  - `g_lr=0.03`, `d_lr=0.1`, `curiosity=0.3` reached `11.4683`.
  - `g_lr=0.01`, `d_lr=0.1`, `curiosity=0` reached `11.9326`.
- Muon/Muon clearly beat AdamW/AdamW in the compact grid. Top results were all Muon/Muon.
- Curiosity helps, but not monotonically. In the large sweep, the best used `0.03`, while the high-LR runner-up used `0.3`.
- Raw score-space curiosity is useful but imperfect. It encourages diversity in score vectors, not necessarily diversity in permutations or edges.
- `LayerNorm` on generator outputs is the strongest normalization choice so far. It prevents pure score-scale drift because only order matters for `argsort`.
- `l2` normalization can work in some high-LR regions, but has worse average performance than `layernorm`.
- No output normalization is clearly worse in this task.
- Bigger MLPs are helpful in the current large sweep. `256,256` produced the best result and a better mean than `128,128`.
- `ranker_tau=16` is the best tested value so far, both by best result and average performance.
- The smaller `batch_size=64` setting found the best run, but `batch_size=128` had a slightly better average over the whole large grid. This needs validation around the winner rather than a broad-grid conclusion.

## Sweep Infrastructure

Use:

```bash
PRESET=compact MAX_JOBS=4 bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh
```

The launcher supports these environment overrides:

```text
OBJECTIVES       e.g. "quantile hybrid"
OPT_PAIRS        e.g. "muon:muon adamw:adamw"
LR_PAIRS         e.g. "0.001:0.01 0.001:0.03 0.01:0.1"
CURIOSITIES      e.g. "0 0.03 0.1 0.3"
HIDDEN_PAIRS     e.g. "128,128:128,128 256,256:256,256"
NORMS            e.g. "layernorm none l2"
TAUS             e.g. "4 8 16"
BATCH_CONFIGS    format "batch_size:buffer_multiplier:ranker_list_size:ranker_sample_pool_size"
CITY_SEEDS       e.g. "0 1 2"
SEEDS            e.g. "0 1 2"
```

Summarize results:

```bash
python examples/tsp/summarize_tsp_results.py \
  --results_dir results/tsp_argsort_lr_wide \
  --top 25 \
  --csv results/tsp_argsort_lr_wide/summary.csv
```

## Next Sweep Axes

Prioritize these axes:

- LR/TTUR around the current winner: `0.001:0.01`, `0.001:0.03`, `0.003:0.03`, `0.003:0.1`, `0.01:0.1`, `0.03:0.1`.
- Curiosity: `0`, `0.03`, `0.1`, `0.3`.
- `ranker_tau`: `4`, `8`, `16`.
- Output norm: `layernorm`, `none`, `l2`.
- Capacity: `128,128`, `256,256`.

Second-stage validation after finding candidates:

- Multiple optimizer seeds.
- Multiple city seeds.
- Batch/buffer scaling such as `64:4:64:256`, `128:4:128:512`, `256:4:256:1024`.

## Known Gap

The current curiosity term is in generator output-score space. For TSP, a route-aware curiosity objective may be better:

- permutation distance from generated routes to elite routes;
- edge-overlap penalty against elite routes;
- novelty over directed or undirected edge sets.

This should be implemented only after the basic LR/tau/norm/capacity behavior is mapped, otherwise it will confound the current baseline.

## Why GFog Is Worse Than TSP Heuristics

Route diagnostics on the current `n=50` best run:

```text
GFog best length:          8.2901
Nearest-neighbor length:   7.2151
2-opt NN length:           6.2945
2-opt applied to GFog:     6.1089
GFog route crossings:      16
2-opt route crossings:     0
GFog / NN edge overlap:    20 / 50 undirected edges
GFog / 2-opt edge overlap: 20 / 50 undirected edges
```

Interpretation:

- GFog is not failing completely. It finds a route with many useful local edges; after 2-opt cleanup, the same route becomes very strong (`6.1089`).
- The main failure is local tour geometry. The learned `argsort` route still contains crossings and avoidable long edges.
- TSP heuristics are explicitly biased toward geometric locality and local edge swaps. GFog currently optimizes global rank of whole permutations through a black-box scalar and has no direct pressure for planarity, edge locality, or 2-opt optimality.
- `argsort(score_vector)` is a weak representation for TSP. Nearby score values define adjacent route positions, but the score space has no native knowledge of Euclidean neighborhoods. Small score changes can reorder globally, and useful local 2-edge swaps are hard to express as smooth moves in score space.
- The discriminator/ranker sees complete score vectors, not edges. It can learn that some whole vectors rank better, but it is not forced to learn reusable local edge features such as "avoid crossing edges" or "prefer short connections".
- The buffer objective is sparse at the structural level: one scalar tour length per full route. It gives no direct credit assignment to individual bad edges or crossings unless D internally discovers it.
- The current curiosity is also score-space novelty, not route/edge novelty. It may maintain proposal diversity without specifically exploring useful edge replacements.

Hypothesis:

GFog is acting as a global proposal generator that can discover decent route neighborhoods, but it lacks the local-improvement operator that makes TSP easy for classical heuristics. Adding a route-aware local search step or an edge-aware objective/representation should close much more of the gap than more generic LR/capacity sweeps.

## GFog With 2-Opt Inside The Objective

`examples/tsp/tsp_argsort.py` now supports:

```text
--objective_two_opt_passes N
```

When `N > 0`, `f` scores `two_opt(argsort(G(z)))` instead of the raw argsort route. The archive stores:

- `best_length`: locally optimized route length used by the black-box objective.
- `best_raw_length`: raw `argsort(G(z))` route length before local search.
- `best_raw_route`: raw generated route.
- `best_route`: 2-opt-improved route.

First run with the current best `n=50` MLP setting and `--objective_two_opt_passes 20`:

```text
best_length:      5.790652
best_raw_length: 28.470009
2opt_random_64:  5.790652
2opt_nn:         6.294477
nearest_neighbor: 7.215113
```

History:

```text
iter 0:    best=5.790652, mean=6.193489
iter 25:   best=5.790652, mean=5.850292
iter 100:  best=5.790652, mean=5.797504
iter 250+: best=5.790652, mean=5.790652
```

Interpretation:

- Putting 2-opt inside `f` makes the final score excellent, but GFog did not improve over a simple 64-start random+2-opt baseline.
- The best score was already present in the initial buffer. Later training mostly collapsed the buffer to routes whose 2-opt cleanup reaches the same local optimum.
- The raw generated route for the best item is very bad (`28.47`), so the learned generator is not directly producing good tours. The local search is doing the useful work.
- This confirms the mechanism diagnosis: local search fixes the problem, but if local search is placed fully inside `f`, it can wash out the learning signal. GFog needs either a representation/objective that learns local edge structure, or a hybrid where GFog proposes seeds and local search is used as a separate postprocessor/selection step rather than the only score.

## Route-Prior Residual Representation

`examples/tsp/tsp_argsort.py` now supports geometric route priors:

```text
--route_prior none|x|y|hilbert|nearest_neighbor
--route_prior_alpha FLOAT
--route_prior_hilbert_bits INT
```

The representation is:

```text
score_i = base_rank_i + alpha * normalized_G(z)_i
route = argsort(score)
```

This preserves the GFog loop but biases the representation toward a geometric base route. Small `alpha` makes local perturbations; large `alpha` recovers freer search.

Base prior route lengths on the default `n=50` city set:

```text
x sort:           19.2199
y sort:           17.3464
hilbert:           7.1136
nearest_neighbor:  7.2151
```

First prior sweep used the current best raw MLP setting:

```text
objective: quantile
G/D: MLP 256,256 / MLP 256,256
optimizer: Muon / Muon
generator_output_norm: layernorm
g_lr: 0.001
d_lr: 0.03
curiosity: 0.03
ranker_tau: 16
batch_size: 64
buffer_multiplier: 4
route_prior: none, x, y, hilbert, nearest_neighbor
route_prior_alpha: 0.03, 0.1, 0.3, 1, 3
```

Top results:

```text
hilbert, alpha=1.0:          6.6615
hilbert, alpha=0.3:          6.7407
hilbert, alpha=3.0:          7.0150
hilbert, alpha=0.03/0.1:     7.1136  (frozen base route)
nearest_neighbor, alpha=1.0: 7.1791
nearest_neighbor, alpha=0.3: 7.1981
free search, no prior:       8.2901
```

Interpretation:

- The inductive bias works. Hilbert prior + residual learning improves the raw no-2opt best from `8.2901` to `6.6615`.
- This is not just returning the fixed prior: Hilbert base route is `7.1136`; alpha `1.0` learns a better route.
- Too-small alpha freezes the base route; too-large alpha weakens the prior and gets worse.
- Hilbert is a better base route than nearest-neighbor for residual learning on this seed, even though both are close as fixed heuristics.
- The result is still above 2-opt from nearest-neighbor (`6.2945`) and 2-opt random starts (`5.7907`), but it closes most of the gap without local search inside `f`.

Refined Hilbert sweep around the first win:

```text
route_prior: hilbert
route_prior_alpha: 0.5, 0.75, 1.0, 1.5, 2.0
ranker_tau: 8, 16, 32
curiosity: 0, 0.01, 0.03, 0.1
```

Best refined result:

```text
hilbert, alpha=1.5, tau=16, curiosity=0.1: best=6.5662
hilbert, alpha=1.5, tau=32, curiosity=0.1: best=6.5662
```

Aggregate pattern:

```text
alpha=1.5: best=6.5662, mean=6.6784
alpha=1.0: best=6.6126, mean=6.6428
alpha=2.0: best=6.6144, mean=6.8317
curiosity=0.1: best=6.5662, mean=6.6973
tau=32: best=6.5662, mean=6.6810
tau=16: best=6.5662, mean=6.7093
```

Current interpretation:

- The best setting shifted to more residual freedom (`alpha=1.5`) and stronger curiosity (`0.1`).
- `alpha=1.0` remains the best average alpha, but `alpha=1.5` gives the best route.
- The refined best `6.5662` is now close to 2-opt NN (`6.2945`) without putting 2-opt inside `f`.

## LSGAN On Hilbert-Prior TSP

LSGAN is now exposed in `examples/tsp/tsp_argsort.py`:

```text
--optimizer lsgan
```

Fair single-setting comparison using the current refined quantile setting except replacing the objective with LSGAN:

```text
optimizer: lsgan
route_prior: hilbert
route_prior_alpha: 1.5
g_lr: 0.001
d_lr: 0.03
curiosity: 0.1
best_length: 6.9886
```

Small LSGAN sweep:

```text
route_prior: hilbert
route_prior_alpha: 1.0, 1.5
g_lr:d_lr:
  0.0003:0.003
  0.001:0.01
  0.001:0.03
  0.003:0.03
  0.003:0.1
  0.01:0.1
curiosity: 0, 0.03, 0.1
```

Best LSGAN result:

```text
alpha=1.0
g_lr=0.001
d_lr=0.03
curiosity=0
best_length=6.7131
final_buffer_mean=6.8541
```

Comparison:

```text
best quantile ranker + Hilbert prior: 6.5662
best LSGAN + Hilbert prior:           6.7131
free raw GFog no prior:               8.2901
nearest neighbor:                     7.2151
2-opt NN:                             6.2945
```

Interpretation:

- LSGAN benefits substantially from the Hilbert prior and beats nearest-neighbor.
- It remains worse than the quantile ranker, even after a small LR/curiosity sweep.
- Curiosity was not helpful for the best LSGAN result; the best used `curiosity=0`.
- LSGAN preferred `alpha=1.0`, while the best quantile run preferred `alpha=1.5`.

## Higher-Dimensional TSP Smoke Runs

The current best `n=50` setting was run unchanged for larger city counts with `1000` iterations:

```text
objective: quantile
G/D: MLP 256,256 / 256,256
generator output norm: layernorm
optimizer: Muon / Muon
g_lr: 0.001
d_lr: 0.03
curiosity: 0.03
ranker_tau: 16
batch_size: 64
buffer_multiplier: 4
```

Results:

```text
n=75:  best=17.4323, random=32.6294, nearest_neighbor=9.3593, 2opt_nn=7.4085
n=100: best=29.1971, random=44.5884, nearest_neighbor=10.1332, 2opt_nn=8.3152
n=150: best=56.7013, random=67.8770, nearest_neighbor=11.2011, 2opt_nn=9.9834
```

Interpretation:

- The method remains better than random search, but scaling degrades badly relative to cheap TSP heuristics.
- Ratio to nearest-neighbor worsens with dimension: `1.86x` at `n=75`, `2.88x` at `n=100`, `5.06x` at `n=150`.
- The `n=50` best setting does not transfer directly. Larger TSP likely needs larger models, larger batch/buffer, different `tau`, and route-aware novelty.

## Set-G / Set-D TSP Sweep

Set-transformer models were wired into `examples/tsp/tsp_argsort.py`:

- `--generator_type set_transformer`: treats the generated batch as a set of latent tokens, applies self-attention across the batch, and emits one route-score vector per token.
- `--discriminator_type set_transformer`: treats the candidate list as a set of score-vector tokens, applies self-attention across candidates, and emits one scalar rank score per candidate.

First `n=50` LR sweep:

```text
objective: quantile
G/D: set_transformer / set_transformer
set_generator_dim: 256
set_discriminator_dim: 256
depth: 2 / 2
heads: 4 / 4
generator output norm: layernorm
optimizer: Muon / Muon
curiosity: 0.03
ranker_tau: 16
batch_size: 64
buffer_multiplier: 4
ranker_list_size: 64
ranker_sample_pool_size: 256
```

LR pairs swept:

```text
0.0003:0.001, 0.0003:0.003, 0.0003:0.01,
0.001:0.001, 0.001:0.003, 0.001:0.01, 0.001:0.03,
0.003:0.003, 0.003:0.01, 0.003:0.03, 0.003:0.1,
0.01:0.01, 0.01:0.03, 0.01:0.1,
0.03:0.03, 0.03:0.1
```

Best result:

```text
g_lr=0.0003
d_lr=0.003
best_length=18.4407
final_buffer_mean=21.1637
artifact: results/tsp_argsort_setset_lr/tsp_city0_n50_quantile_gtypeset_transformer_dtypeset_transformer_gh256x256_dh256x256_sg256x2h4_sd256x2h4_b64_bm4_rank64_pool256_tau16_normlayernorm_goptmuon_doptmuon_glr0.0003_dlr0.003_curio0.03_batch_ct2_iter1000_seed0.npz
```

Interpretation:

- This direct Set-G + Set-D formulation is much worse than the MLP best (`8.2901`).
- Most LR settings stayed around `19-21`, only marginally better than random search (`20.4142`) and far worse than nearest-neighbor (`7.2151`).
- The model is also much slower than MLP. The direct set approach is not worth scaling without changing the mechanism.
- Likely issue: batch self-attention couples proposals, but the objective gives independent black-box scores. This can reduce useful independent exploration unless the architecture has a clearer recombination/selection mechanism or route-aware diversity objective.
