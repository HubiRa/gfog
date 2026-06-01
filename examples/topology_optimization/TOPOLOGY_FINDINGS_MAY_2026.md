# Topology Optimization Findings, May 2026

This note captures the current practical state of the topology optimization
experiments so the results survive across sessions.

## Current Best 40x20 Baseline

The best 40x20 setting is the fixed-material sorting formulation:

```text
encoding=sorted_material
sorted_material_profile=binary
density_filter_radius=0
projection_beta=0
generator_type=conv
generator_output_norm=centered_l2
discriminator_type=mlp
optimizer_type=quantile_ranked_default
ranker_target_curve=exp
ranker_tau=4
ranker_weight=1
ranker_steps=1
ranker_sample_pool_size=128
ranker_sample_mode=random_top_pool
g_torch_optimizer=muon
d_torch_optimizer=muon
g_lr=0.03
d_lr=0.1
batch_size=64
buffer_multiplier=8
curiosity_space=raw
curiosity_reference=buffer
curiosity=0.0003
```

Best results:

```text
3k seed 0:  best_feasible_compliance = 72.5222
3k seed 1:  best_feasible_compliance = 78.1218
3k seed 2:  best_feasible_compliance = 74.2828
```

This beats the previous 3k no-normalization/no-curiosity baseline on all three
seeds:

```text
old 3k seed 0: 73.6800
old 3k seed 1: 78.8317
old 3k seed 2: 74.9235
```

It also improves the best individual 40x20 result from `72.9039` at 10k to
`72.5222` at 3k. The older `topk_volume + density_filter_radius=1 +
projection_beta=1` result was `90.7142` at 10k.

Interpretation: fixing the material amount/distribution and letting `G` only
learn the ordering/placement is a strong representation. The binary material
histogram beats smooth `linear` and `sigmoid` sorted histograms in the first
tests. The important later correction is to normalize the generated
sorted-material genome with centered L2 before `D`, `f`, and curiosity. This
keeps distances meaningful and makes raw uniformity usable.

## Post-TSP Lesson Applied To Topopt

The TSP experiments showed that GFog needs a representation with meaningful
local neighborhoods. For topology optimization this means:

- Keep a spatial generator prior. The Conv decoder is still the best practical
  low-effort prior.
- Avoid flat high-dimensional genomes unless `f` adds strong locality.
- Keep constraints and deterministic projection/repair inside `f`, not inside
  trainable model layers.
- Track full learning curves, not only final best values.
- Use one ranker update per iteration; tune learning rates instead of increasing
  ranker steps.

## Mixed Evaluated Ranker Test

We tested the cleaner mixed evaluated ranking idea:

```text
optimizer_type=hybrid_contextual_utility
```

This trains `D` on mixed sorted lists containing buffer entries plus recently
evaluated `G` outputs. This is conceptually cleaner than training `D(fake)->0`
because it keeps `D` connected to the distribution that `G` actually visits.

1k results:

```text
mixed_ranker_connectivity: 101.574
mixed_ranker_curiosity:    101.945
quantile_best:             102.414
mixed_ranker:              163.332
```

3k results:

```text
quantile_best:             74.2849
mixed_ranker_curiosity:    74.4985
mixed_ranker_connectivity: 83.1028
```

Interpretation: the mixed ranker is not currently better than quantile. Small
raw batch curiosity makes it competitive, but the known quantile objective
remains the baseline. The connectivity objective helped at 1k but hurt at 3k.

Artifacts:

```text
results/topopt_spatial_lessons_1k
results/topopt_spatial_lessons_3k
```

## Force Distribution Robustness

Previously most runs used one right-edge center point load. We added same-grid
load cases to test whether the setup is overfit to that single force vector:

```text
center_point
right_top_point
right_bottom_point
right_two_points
right_edge_uniform
right_edge_shear
```

Code:

```text
examples/topology_optimization/cantilever_fem.py
examples/topology_optimization/run_topopt_loadcase_suite.sh
```

1k quantile baseline results:

```text
right_edge_shear:     7.0117
right_top_point:     84.4268
right_edge_uniform:  88.3202
center_point:       102.4140
right_two_points:   103.4320
right_bottom_point: 127.0420
```

Do not compare absolute compliance directly across load cases; load direction
and solid-compliance scales differ. Compare relative progress curves and design
behavior within each load case.

All six load-case runs saved traces:

```text
history shape: (41, 12)
checkpoint interval: 25 iterations
eval_count: 512 -> 64512
```

History columns:

```text
iteration
eval_count
best_last
best_feasible_last
mean_last
median_last
p10_last
p90_last
feasible_rate
mean_volume_violation
mean_roughness_violation
mean_connectivity_violation
```

Artifacts:

```text
results/topopt_loadcase_suite_1k
```

Summarize with:

```bash
python examples/topology_optimization/summarize_topopt_results.py results/topopt_loadcase_suite_1k
```

## Reduced-Volume 40x20 Benchmark (`volume_max=0.35`)

We reduced the material budget from the earlier `0.48` benchmark to `0.35`.
This made the problem meaningfully harder and exposed LR/seed sensitivity.

Reference baselines:

```text
continuous SIMP/OC, no filter: 107.4058
binarized OC reference:        about 110.56
best GFog run so far:          109.3426
```

The best GFog result is close to the continuous OC solution and slightly better
than the binarized OC reference, but it is not a robust average result yet.

Best individual GFog artifact:

```text
results/topopt_vol035_lr_seed_neighborhood_1k/
  quantile_ranked_default_tau4_iter1000_vol0.35_bs128_bufx4_...
  _glr0.07_dlr0.12_curio0.0003_..._seed2/
  top_designs_curiosity_0.0003_seed_2.npz
```

Best individual setting:

```text
optimizer_type=quantile_ranked_default
encoding=sorted_material
sorted_material_profile=binary
generator_type=conv
generator_output_norm=centered_l2
discriminator_type=mlp
g_torch_optimizer=muon
d_torch_optimizer=muon
g_lr=0.07
d_lr=0.12
batch_size=128
buffer_multiplier=4
ranker_target_curve=exp
ranker_tau=4
ranker_list_size=64
ranker_sample_pool_size=128
ranker_steps=1
curiosity=0.0003
curiosity_space=raw
curiosity_reference=buffer
seed=2
n_iter=1000
best_feasible_compliance=109.3426
```

The most robust LR neighborhood in the 3-seed, 1k sweep was not the sharp
single-seed best:

```text
g_lr=0.07, d_lr=0.08:
  seeds: 124.00 / 113.11 / 117.84
  mean best: 118.32

g_lr=0.07, d_lr=0.12:
  seeds: 127.76 / 121.42 / 109.34
  mean best: 119.51

g_lr=0.06, d_lr=0.10:
  seeds: 114.65 / 524.93 / 135.20
  mean best: 258.26
```

Interpretation: the old `g_lr=0.06, d_lr=0.10` run produced a good seed-0
number but is not stable. For further work at `volume_max=0.35`, use
`g_lr=0.07, d_lr=0.08` as the robust baseline, and keep `g_lr=0.07,
d_lr=0.12` as the high-upside candidate.

## Negative Robustness Results

Several plausible D-side robustness changes made results worse:

```text
D-only value auxiliary head:
  utility_weight=0.001: 405.76
  utility_weight=0.003: 114.20
  utility_weight=0.01:  124.96
  utility_weight=0.1:   579.96

ranker_fake_weight with one rank list:
  fakew=0:   2422.17
  fakew=0.1: 2422.17
  fakew=0.3: 1619.97
  fakew=1:    114.65

ranker_fake_repeats:
  frep=1: 114.65
  frep=2: 130.14
  frep=4: 117.26
  frep=8: 170.50

ranker_list_repeats=4:
  fakew=1: 129.78
  lower fake weights failed badly
```

Lessons:

- G must not receive direct compliance/value supervision in this formulation.
  Keep G on the rank/fake reward signal only.
- A shared value head on D is not a free stabilizer. Absolute or local value
  regression can conflict with top-buffer discrimination in the shared trunk.
- The `D(fake)=0` term is essential. Removing or weakening it lets G exploit
  uncalibrated high-score holes away from the buffer manifold.
- More fake batches per D update did not help; it changes the rank/fake balance
  and can over-regularize.
- Averaging multiple rank lists per D update also did not help in this setting.
- The useful robustness lever so far is TTUR-style LR tuning, not extra D
  objectives or more D steps.

## OC Comparison

For the simple compliance-minimization cantilever with one volume constraint,
SIMP/OC remains a very strong baseline. Our GFog binary result is visually
reasonable and close to OC, but OC has the advantage of using analytic
sensitivities and a problem-specific update rule.

GFog becomes more interesting when the objective or constraints are hard to
differentiate, discontinuous, expensive to hand-code, or not naturally handled
by a closed-form OC update. See the "OC-like method limits" section below for
candidate problem classes.

## OC-Like Method Limits

Vanilla OC works well for classic density-based compliance minimization with a
global volume constraint because the KKT conditions yield a simple monotone
update. It is much less natural for:

- black-box objectives where adjoint/sensitivity code is unavailable;
- discrete objectives or projections, such as exact binary material placement,
  connectivity postprocessing, integer components, lattice catalog choices, or
  manufacturing rules with hard if/then logic;
- stress-constrained designs with many local constraints and singular stress
  behavior;
- buckling, eigenfrequency, contact, friction, large deformation, plasticity,
  fracture, fatigue, crash, and transient dynamics;
- robust or stochastic topology optimization over many uncertain load/material
  cases when the aggregate objective is non-smooth or simulation-defined;
- multiphysics inverse design where the objective is evaluated by an external
  simulator and gradients are unavailable or unreliable;
- learned or human-defined objectives, for example aesthetics, manufacturability
  scores, classifier scores, or downstream performance measured by another
  black box.

Many of these can still be solved with gradient-based methods if good
sensitivities exist, usually MMA/SQP/adjoint methods rather than simple OC. The
GFog use case is strongest when `f` is cheap enough to evaluate in large
batches but awkward, non-differentiable, or expensive to differentiate.

## Candidate Non-OC-Friendly Benchmarks

Immediate local benchmarks:

- Robust multi-load cantilever: pass `--robust_load_cases ...` to evaluate each
  candidate topology under several existing force distributions and rank by
  `--robust_load_aggregate max|mean|cvar`. This keeps the FEM backend local but
  makes `f` less like the vanilla OC example because the objective is a
  non-smooth aggregate of multiple simulations.
- Connectivity/manufacturing penalties: use `--connectivity_max` or
  connectivity ladders to prepend a hard black-box connected-material objective
  before compliance. The implementation already measures material disconnected
  from the left support.
- Future local addition: add overhang/support-count/component-count penalties
  on the decoded binary image. These are cheap image operations and a better
  GFog fit than continuous compliance alone.

External benchmarks worth trying later:

- PyTOPress: Python/NumPy/SciPy code for topology optimization with
  design-dependent pressure loads. This is attractive because the load changes
  with the topology, making it a better black-box benchmark than a fixed-force
  cantilever.
- pyMOTO examples: stress constraints, overhang filters, robust formulations,
  eigenfrequency, transient thermal, self-weight, and compliant mechanisms.
  These are gradient-aware internally, but can be wrapped as black-box `f`
  objectives for GFog comparisons.

First robust multi-load sanity run:

```text
40x20, volume_max=0.35, sorted binary, Conv G / MLP D, Muon/Muon
robust_load_cases = center_point right_top_point right_bottom_point right_edge_uniform
robust_load_aggregate = max
n_iter = 1000, batch_size = 64, buffer_multiplier = 4
best aggregate compliance = 372.1453
mean top-9 aggregate compliance = 392.1672
mean_hamming top-9 = 0.0386
runtime = 2:54 local
```

Best design individual compliances:

```text
center_point        348.9781
right_top_point     372.1453
right_bottom_point  363.2189
right_edge_uniform  351.4646
```

This is not yet tuned and is much harder than the single-load benchmark, but it
confirms the robust objective works and produces a balanced max-load design
rather than optimizing only the center-point cantilever.

## Removal-Ladder Objective

Implemented an opt-in staged material-removal objective:

```bash
bash examples/topology_optimization/run_topopt_removal_ladder.sh
```

The generator emits one priority score per element. The black-box objective
keeps the top-k elements at each configured material fraction and returns
lexicographic compliance-threshold violations, followed by the usual
`volume_violation, roughness_violation, compliance` summary tail for plotting.
Default first test:

```text
volumes:      0.50 0.45 0.40 0.35 0.30
thresholds:  80   95   115  145  190
load_case:   center_point
encoding:    sorted_material binary
G/D:         Conv G / MLP D, Muon/Muon
```

The robust-load LR scale `g_lr=0.01, d_lr=0.1` was poor: after 500 iterations
it did not solve even the `0.50` material stage. The single-load LR scale
`g_lr=0.07, d_lr=0.08` worked better. At 1k iterations it produced:

```text
archive value vector:
[0.0, 0.0, 0.0, 98.7017, 5200.7251, 0.0, 0.0, 5390.7251]

same priority field by volume:
vol=0.50  C=78.2 / 80
vol=0.45  C=92.6 / 95
vol=0.40  C=113.6 / 115
vol=0.35  C=243.7 / 145
vol=0.30  C=5390.7 / 190
```

Interpretation: the staged-removal formulation works mechanically and gives a
meaningful nested ordering signal, but the first run only learned an ordering
that survives down to 40% material. The load path breaks at 35%, so the next
run should either extend training, tune LR around `g=0.07,d=0.08`, or use a
shorter/adaptive ladder before pushing to 30% material.

Stage-level connectivity tests:

- Added `--removal_ladder_connectivity_max`. When set, every removal stage gets
  a disconnected-material violation before its compliance violation.
- `--removal_ladder_connectivity_max 0` with `g=0.07,d=0.08`, 500 iters was
  poor. It connected the first stage but did not produce competitive
  compliance: first entries were `[0.0, 68.5, 0.0389, 313.7, ...]`, so the
  objective was dominated by connectivity/compliance failures very early.
- Added stage-wise `--binhead_connect_support` repair for removal ladders, wired
  through `REMOVAL_CONNECT_REPAIR=true`. This guarantees the evaluated stage
  design is support-connected after repair, but it is currently too slow and
  poor: a small `100` iter, `bs=32` run took `3:28` and failed the first
  compliance stage badly.

Current take: do not use stage-wise repair as the default unless the repair is
optimized. For now, the more promising path is a shorter/adaptive ladder and
possibly a mild connectivity penalty, not hard repair at every stage.

Adaptive 35% ladder result:

```text
volumes:      0.50 0.45 0.40 0.35
thresholds:  80   95   125  220
setting:     g_lr=0.07, d_lr=0.08, 1k iters, bs=64
archive value vector:
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 151.5835]

same priority field by volume:
vol=0.50  C=78.4 / 80
vol=0.45  C=92.6 / 95
vol=0.40  C=115.6 / 125
vol=0.35  C=151.6 / 220
```

This is the best removal-ladder result so far. The key change was relaxing the
40% cap from 115 to 125 and making the 35% cap loose enough to enter the final
tie-break. This supports the adaptive-threshold hypothesis: stages should be
reachable first, then tightened.

Tightening only the final 35% cap from 220 to 180 and then 160 did not change
the best result; both reruns returned the same `C35=151.5835` solution with all
stage violations zero. This suggests the current bottleneck is not the final
threshold value once it is reachable. Next useful move is structural: add an
intermediate lower-volume stage such as `0.325`, or run longer/multi-seed,
rather than only tightening the `0.35` cap.

Direct fixed-volume comparison on the same `volume_max=0.35`, center-point task:

```text
setting: usual quantile_ranked_default, sorted binary, Conv G / MLP D,
         Muon/Muon, g_lr=0.07, d_lr=0.08, curiosity=0.0003,
         batch_size=64, buffer_multiplier=4, 1k iters

seed0: C35=128.4664
seed1: C35=529.5534
seed2: C35=413.1860
```

Interpretation: direct fixed-volume optimization has better upside than the
removal ladder (`128.5` vs `151.6`) but is much less stable across seeds in
this 1k comparison. The removal ladder currently looks more like a nested
ordering/robustness experiment than a pure fixed-volume performance win.

Degenerate compliance-only test with the official `Levels.ladder` mechanism:

```text
setting: same as above, plus --levels_ladder compliance:180,160,140
         --levels_ladder_final_open compliance

seed0: C35=126.0838, mean top9=126.7317
seed1: C35=529.5534, mean top9=532.2278
seed2: C35=413.1860, mean top9=422.7203
```

This improved the already-good seed slightly (`128.47 -> 126.08`) but did not
fix bad seeds. Mechanistically, a compliance-only ladder is almost monotone
equivalent to raw compliance: `[max(C-180,0), max(C-160,0), max(C-140,0), C]`
preserves the same order for most comparisons. It is therefore not a real
robustness mechanism by itself. Ladders become interesting only when they add a
different staged signal, e.g. changing volume/material constraints or mixing
objectives.

Actual material+compliance ladder on the usual Conv-G/MLP-D ranker setup:

```text
encoding=direct
hard_binarize=true
G=conv, D=mlp, Muon/Muon
g_lr=0.07, d_lr=0.08
batch_size=64, buffer_multiplier=4
curiosity=0.0003
LEVELS_LADDER="volume:min:0.50,0.45,0.40,0.35 compliance:min:300,260,220,180"
LEVELS_LADDER_FINAL_OPEN=compliance

centered_l2 G norm, 500 iters:
  top volumes: about 0.42-0.45
  no feasible volume<=0.35 sample
  best_any_compliance=222.8466

l2 G norm, 500 iters:
  top volumes: about 0.3425-0.3500
  best_feasible_compliance=155.0014

l2 G norm, 1000 iters:
  top volumes: about 0.3438-0.3500
  best_feasible_compliance=151.8213
  mean top9=153.4096
```

This is the intended ladder class: material/volume rungs and compliance rungs
are interleaved. It is not yet competitive with fixed sorted-binary
optimization, but it is a useful finding. A material ladder needs a global
material-control direction in the genome. `centered_l2` removes that direction;
plain `l2` keeps outputs scale-bounded while still allowing the sign/mean
distribution to change material usage.

Coarse top-k symmetry-reduction test:

```text
Goal: reduce sorted/top-k permutation symmetry by having G emit a lower-res
score grid, bilinearly upsampling it to 40x20, then applying the usual full-grid
top-k binary projection.

common setting:
  encoding=coarse_topk_volume
  volume_max=0.35
  Conv G / MLP D, Muon/Muon
  centered_l2 G output norm
  batch_size=64, buffer_multiplier=4
  curiosity=0.0003
  n_iter=1000

20x10 code grid, g_lr=0.07, d_lr=0.08:
  best_feasible_compliance=486.9804
  mean top9=487.8121
  mean_hamming=0.0068

10x5 code grid, g_lr=0.07, d_lr=0.08:
  best_feasible_compliance=549.5952
  mean top9=552.8832
  mean_hamming=0.0079

20x10 code grid, g_lr=0.03, d_lr=0.10:
  best_feasible_compliance=521.5104
  mean top9=524.8285
  mean_hamming=0.0150
```

This drop-in encoding is a negative result. It reduces the rank-symmetry, but
the bilinear-upsample + top-k projection over-biases the search toward smooth
single-blob threshold bands. The best visual result was a horizontal material
band, not a diagonal/truss-like load path. If revisiting this direction, do not
use plain coarse top-k as the default. A better variant would be coarse-to-fine:
coarse spatial prior plus a fine residual/refinement channel, patch-level
selection followed by intra-patch refinement, or a multistage ladder that first
solves coarse structure and then releases fine degrees of freedom.

Non-sorting encoding sweep:

```text
common:
  volume_max=0.35
  quantile_ranked_default, tau=4
  batch_size=64, buffer_multiplier=4
  curiosity=0.0003
  n_iter=1000, seed=0

soft_volume, layernorm, projection_beta=0:
  exact soft volume, gray densities
  best_feasible_compliance=386.6098
  mean top9=387.5450

soft_volume, layernorm, projection_beta=8:
  exact soft volume plus projection sharpening
  best_feasible_compliance=217.4709
  mean top9=218.3932

soft_volume, layernorm, hard_binarize=true:
  adaptive threshold / approximate level-set projection
  best_feasible_compliance=247.2338
  mean top9=252.6333

coarse level-set ladder:
  encoding=coarse, code_grid=20x10, hard_binarize=true, l2 G norm
  LEVELS_LADDER="volume:min:0.50,0.45,0.40,0.35 compliance:min:300,260,220,180"
  no feasible volume<=0.35 sample
  best_any_compliance=275.3018

bar primitives, 16 bars:
  encoding=bar_primitives
  G=mlp, D=mlp, no G output norm
  bar_count=16, width=[0.015,0.06], edge_softness=0.008
  g_lr=0.01, d_lr=0.03
  same material/compliance ladder as above
  best_feasible_compliance=158.2036
  mean top9=161.4743
  mean_hamming=0.1265

bar primitives, 24 thinner bars:
  best_feasible_compliance=183.3318
  mean top9=188.3109

bar primitives, 16 bars, hard_binarize=true:
  no fully feasible sample because roughness violation stayed nonzero
  best_any_compliance=200.3494
```

16-bar LR sweep, same material/compliance ladder:

```text
g_lr   d_lr   best_feasible   best_any   mean_top9   feasible_rate
0.003  0.01   nan             383.0736   671.6476    0.000
0.003  0.03   nan             197.9917   209.1115    0.000
0.003  0.10   183.2875        183.2875   192.7215    1.000
0.010  0.01   210.8403        210.8403   214.0112    1.000
0.010  0.03   158.2036        158.2036   161.4743    1.000
0.010  0.10   nan             194.5181   206.8139    0.000
0.030  0.01   159.1361        159.1361   159.9729    1.000
0.030  0.03   208.1070        208.1070   209.6070    1.000
0.030  0.10   196.6062        176.6600   202.2141    0.444
```

Conclusion: the simple soft-volume replacements remove sorting but are not
competitive. Projection sharpening helps, but still loses badly to sorted
binary and to direct hard-binary material ladders. The first structured
primitive encoding is more interesting: 16 differentiable-in-parameter bars
rasterized inside `f` produced a plausible truss-like design at `158.2`,
close to the direct material ladder (`151.8`) and much better than soft-volume.
It is still far behind fixed sorted binary, but it is the first non-sorting
representation here with a useful topology prior instead of a per-cell ranking
problem. The LR sweep did not improve over `g_lr=0.01,d_lr=0.03`; `g_lr=0.03,
d_lr=0.01` was almost tied but collapsed diversity. The next primitive tests
should tune bar anchors/constraints rather than simply increasing bar count.

## New Utilities

Experiment suites:

```text
examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
examples/topology_optimization/run_topopt_loadcase_suite.sh
examples/topology_optimization/run_topopt_robust_multiload.sh
examples/topology_optimization/run_topopt_removal_ladder.sh
```

`run_topopt_ttur_sweep.sh` also supports official ladders and variable
encodings:

```text
LEVELS_LADDER="compliance:180,160,140" LEVELS_LADDER_FINAL_OPEN=compliance \
  bash examples/topology_optimization/run_topopt_ttur_sweep.sh

ENCODING=direct HARD_BINARIZE=true GENERATOR_OUTPUT_NORM=l2 \
LEVELS_LADDER="volume:min:0.50,0.45,0.40,0.35 compliance:min:300,260,220,180" \
LEVELS_LADDER_FINAL_OPEN=compliance RUN_TAG=mat_comp_ladder \
  bash examples/topology_optimization/run_topopt_ttur_sweep.sh

ENCODING=coarse_topk_volume COARSE_GRID_WIDTH=20 COARSE_GRID_HEIGHT=10 \
RUN_TAG=coarse20x10 \
  bash examples/topology_optimization/run_topopt_ttur_sweep.sh

ENCODING=bar_primitives GENERATOR_TYPE=mlp GENERATOR_OUTPUT_NORM=none \
BAR_COUNT=16 BAR_WIDTH_MIN=0.015 BAR_WIDTH_MAX=0.06 BAR_EDGE_SOFTNESS=0.008 \
LEVELS_LADDER="volume:min:0.50,0.45,0.40,0.35 compliance:min:300,260,220,180" \
LEVELS_LADDER_FINAL_OPEN=compliance RUN_TAG=bars16_ladder \
  bash examples/topology_optimization/run_topopt_ttur_sweep.sh
```

Summarizer:

```text
examples/topology_optimization/summarize_topopt_results.py
```

`cantilever_fem.py` now saves richer history telemetry in every `.npz`
artifact, including feasible-rate and constraint-violation traces.

## Current Recommendation

Use this as the default 40x20 baseline:

```bash
N_ITER=3000 FEM_WORKERS=8 \
EXPERIMENTS="quantile_best" \
OUTPUT_ROOT=results/topopt_baseline_3k \
bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
```

The helper script now defaults `quantile_best` to:

```text
generator_output_norm=centered_l2
d_lr=0.1
curiosity=0.0003
curiosity_space=raw
curiosity_reference=buffer
```

For robustness, run the force suite:

```bash
N_ITER=1000 FEM_WORKERS=8 \
OUTPUT_ROOT=results/topopt_loadcase_suite_1k \
bash examples/topology_optimization/run_topopt_loadcase_suite.sh
```

For method development, compare against:

```bash
N_ITER=3000 FEM_WORKERS=8 \
EXPERIMENTS="quantile_best mixed_ranker_curiosity" \
OUTPUT_ROOT=results/topopt_ranker_compare_3k \
bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
```

The bar to beat is currently `74.2849` at 3k and `72.9039` at 10k on the
40x20 sorted-binary center-load benchmark.
