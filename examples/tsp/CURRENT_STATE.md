# TSP GFog Current State

This is the compact handoff for the current TSP work. Detailed run notes live in `examples/tsp/TSP_EXPERIMENT_LEARNINGS.md`.

## Current Code

Main files:

- `examples/tsp/tsp_argsort.py`
- `examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh`
- `examples/tsp/summarize_tsp_results.py`
- `examples/tsp/TSP_EXPERIMENT_LEARNINGS.md`

The TSP objective is:

```text
G(z) -> score vector of length n_cities
route = argsort(score)
f(route) = Euclidean tour length
```

Lower is better.

Implemented model options:

- `--generator_type mlp`
- `--generator_type set_transformer`
- `--discriminator_type mlp`
- `--discriminator_type set_transformer`

Implemented objective option:

- `--objective_two_opt_passes N`

Implemented optimizer options:

- `--optimizer quantile`
- `--optimizer hybrid`
- `--optimizer lsgan`

When `N > 0`, `f` scores `two_opt(argsort(G(z)))` and archives both raw and optimized routes:

- `best_raw_route`, `best_raw_length`
- `best_route`, `best_length`

## Best Raw Argsort Result

Best result without local search inside `f` and without a geometric prior:

```text
n_cities: 50
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
ranker_list_size: 64
ranker_sample_pool_size: 256
n_iter: 1000
best_length: 8.2901
final_buffer_mean: 8.5886
```

Artifact:

```text
results/tsp_argsort_large_sweep/tsp_city0_n50_quantile_gh256x256_dh256x256_b64_bm4_rank64_pool256_tau16_normlayernorm_goptmuon_doptmuon_glr0.001_dlr0.03_curio0.03_batch_ct2_iter1000_seed0.npz
```

Baselines on the same city set:

```text
random search best, matched budget: 20.4142
nearest neighbor:                    7.2151
2-opt from nearest neighbor:          6.2945
2-opt random starts:                  5.7907
```

So GFog is much better than random search but still worse than cheap TSP heuristics.

## Best Route-Prior Result

Route-prior residual scoring is implemented:

```text
score_i = base_rank_i + alpha * normalized_G(z)_i
route = argsort(score)
```

CLI:

```text
--route_prior none|x|y|hilbert|nearest_neighbor
--route_prior_alpha FLOAT
--route_prior_hilbert_bits INT
```

Best result so far:

```text
n_cities: 50
route_prior: hilbert
route_prior_alpha: 1.5
objective: quantile
G/D: MLP 256,256 / MLP 256,256
optimizer: Muon / Muon
generator_output_norm: layernorm
g_lr: 0.001
d_lr: 0.03
curiosity: 0.1
ranker_tau: 16
batch_size: 64
buffer_multiplier: 4
ranker_list_size: 64
ranker_sample_pool_size: 256
n_iter: 1000
best_length: 6.5662
```

Base route lengths:

```text
x sort:           19.2199
y sort:           17.3464
hilbert:           7.1136
nearest_neighbor:  7.2151
```

This is a real learned improvement over the Hilbert base route, not just the fixed prior. It improves substantially over free GFog (`8.2901`) and beats nearest-neighbor (`7.2151`), but it is still worse than 2-opt from nearest-neighbor (`6.2945`) and 2-opt random starts (`5.7907`).

## Important Sweep Findings

Strongest knobs:

- `D` must learn materially faster than `G`.
- Muon/Muon is better than AdamW/AdamW in tested grids.
- `layernorm` on generator outputs is important.
- `ranker_tau=16` is best among `4, 8, 16`.
- `256,256` MLPs beat `128,128` in the large sweep.
- Curiosity helps but is non-monotonic; best raw run used `0.03`.

Large sweep top region:

```text
g_lr=0.001, d_lr=0.03, curiosity=0.03, layernorm, tau=16
g_lr=0.003, d_lr=0.1,  curiosity=0.3,  layernorm, tau=16
```

Longer run:

```text
same best setting, n_iter=5000
best stayed 8.2901
buffer mean tightened from 8.5886 to 8.5505
```

Longer training alone did not improve the best route.

## Diagnostics

The current best GFog route has bad local geometry:

```text
GFog best length:          8.2901
2-opt applied to GFog:     6.1089
GFog route crossings:      16
2-opt route crossings:     0
GFog / NN edge overlap:    20 / 50 undirected edges
GFog / 2-opt edge overlap: 20 / 50 undirected edges
```

Interpretation:

- GFog finds useful route neighborhoods.
- The raw `argsort` representation does not clean up crossings or local edge swaps.
- Classical TSP heuristics win because they have the right local geometric inductive bias.

## 2-Opt Inside The Objective

Run with `--objective_two_opt_passes 20` using the current best MLP setting:

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

Conclusion:

- Putting full 2-opt inside `f` makes the score excellent.
- But GFog did not beat random+2-opt; the best was already in the initial buffer.
- Raw generated route quality was poor, so local search did almost all useful work.
- Full local search inside `f` washes out the learning signal.

## Higher-Dimensional Smoke Runs

The best `n=50` setting was run unchanged:

```text
n=75:  best=17.4323, random=32.6294, nearest_neighbor=9.3593, 2opt_nn=7.4085
n=100: best=29.1971, random=44.5884, nearest_neighbor=10.1332, 2opt_nn=8.3152
n=150: best=56.7013, random=67.8770, nearest_neighbor=11.2011, 2opt_nn=9.9834
```

Scaling degrades badly relative to heuristics. The `n=50` setting does not transfer directly.

## Set-G / Set-D Result

Set transformer G/D were implemented and swept on `n=50`.

Best Set-G + Set-D result:

```text
G/D: set_transformer / set_transformer
dim: 256 / 256
depth: 2 / 2
heads: 4 / 4
optimizer: Muon / Muon
curiosity: 0.03
tau: 16
batch: 64
best_length: 18.4407
g_lr: 0.0003
d_lr: 0.003
```

Conclusion:

- Direct Set-G + Set-D is much worse than MLP.
- It is only slightly better than random search and much slower.
- Batch self-attention does not help unless there is a better recombination/local-search mechanism.

## LSGAN Result

LSGAN has been compared on the Hilbert-prior setup.

Best LSGAN sweep result:

```text
optimizer: lsgan
route_prior: hilbert
route_prior_alpha: 1.0
G/D: MLP 256,256 / MLP 256,256
optimizerG/optimizerD: Muon / Muon
g_lr: 0.001
d_lr: 0.03
curiosity: 0
batch_size: 64
buffer_multiplier: 4
best_length: 6.7131
final_buffer_mean: 6.8541
```

Comparison:

```text
best quantile + Hilbert prior: 6.5662
best LSGAN + Hilbert prior:    6.7131
free raw GFog no prior:        8.2901
nearest neighbor:              7.2151
2-opt NN:                      6.2945
```

Conclusion:

- LSGAN is useful once the Hilbert prior exists and beats nearest-neighbor.
- Quantile ranker still wins.
- LSGAN preferred `alpha=1.0` and no curiosity, unlike quantile where the best was `alpha=1.5`, `curiosity=0.1`.

## Current Hypothesis

The bottleneck is representation and inductive bias, not just optimizer tuning.

The current `argsort(score_vector)` formulation can express any route, but it does not make good TSP structure easy:

- short edges are not built in;
- crossing avoidance is not built in;
- local 2-opt moves are not simple smooth changes in score space;
- D sees whole score vectors, not explicit edges;
- curiosity is score-space novelty, not route/edge novelty.

GFog is acting as a global proposal generator. It can find decent basins, but it lacks a local geometric bias.

The route-prior experiment supports this: adding a Hilbert geometric prior improved the raw route from `8.2901` to `6.5662`.

## Recommended Next Experiment

Validate and refine the current Hilbert-prior best:

```text
route_prior: hilbert
alpha: around 1.0-1.75
tau: 16, 32
curiosity: around 0.03-0.2
```

Then validate across seeds/city seeds.

Other follow-up candidates:

- Route-aware curiosity based on edge overlap.
- Hybrid objective that tracks both raw route length and post-2opt route length.
- Edge-aware D or auxiliary crossing/edge-length features.
- Higher-dimensional TSP retest with Hilbert residual prior.

Run current best prior setting:

```bash
PRESET=compact MAX_JOBS=1 N_ITER=1000 \
OUTPUT_DIR=results/tsp_argsort_hilbert_best_repeat \
OBJECTIVES="quantile" \
OPT_PAIRS="muon:muon" \
LR_PAIRS="0.001:0.03" \
CURIOSITIES="0.1" \
HIDDEN_PAIRS="256,256:256,256" \
NORMS="layernorm" \
TAUS="16" \
BATCH_CONFIGS="64:4:64:256" \
ROUTE_PRIORS="hilbert" \
ROUTE_PRIOR_ALPHAS="1.5" \
bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh
```

## Useful Commands

Summarize a result directory:

```bash
python examples/tsp/summarize_tsp_results.py \
  --results_dir results/tsp_argsort_large_sweep \
  --top 25 \
  --csv results/tsp_argsort_large_sweep/summary.csv
```

Run current best raw setting:

```bash
PRESET=compact MAX_JOBS=1 N_ITER=1000 \
OUTPUT_DIR=results/tsp_argsort_best_repeat \
OBJECTIVES="quantile" \
OPT_PAIRS="muon:muon" \
LR_PAIRS="0.001:0.03" \
CURIOSITIES="0.03" \
HIDDEN_PAIRS="256,256:256,256" \
NORMS="layernorm" \
TAUS="16" \
BATCH_CONFIGS="64:4:64:256" \
bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh
```

Run current best with 2-opt inside `f`:

```bash
PRESET=compact MAX_JOBS=1 N_ITER=1000 \
OUTPUT_DIR=results/tsp_argsort_f2opt_repeat \
OBJECTIVE_TWO_OPT_PASSES=20 \
OBJECTIVES="quantile" \
OPT_PAIRS="muon:muon" \
LR_PAIRS="0.001:0.03" \
CURIOSITIES="0.03" \
HIDDEN_PAIRS="256,256:256,256" \
NORMS="layernorm" \
TAUS="16" \
BATCH_CONFIGS="64:4:64:256" \
bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh
```
