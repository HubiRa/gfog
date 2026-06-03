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

Smoothed rerun of this baseline:

```text
artifact:
  results/topopt_best40_sorted_material_centered_l2_smoothed_3k_seed0/top_designs_curiosity_0.0003_seed_0.npz

same current-best baseline, except:
  density_filter_radius=1

3k seed 0:
  best_feasible_compliance=94.3184
  mean top9 compliance=94.4403
  mean_hamming=0.0047
  intermediate-density pixel count in top9=2822
```

Blob-initialized rerun of this baseline:

```text
artifact:
  results/topopt_best40_sorted_material_centered_l2_blobinit_3k_seed0/top_designs_curiosity_0.0003_seed_0.npz

same current-best baseline, except:
  initial_buffer_mode=random_blobs
  initial_blob_count_max=8
  initial_blob_radius_min=0.04
  initial_blob_radius_max=0.30
  initial_blob_min_hamming=0.2

3k seed 0:
  best_feasible_compliance=76.4370
  mean top9 compliance=76.4423
  feasible_rate=1.0
  mean_hamming=0.0011
  intermediate-density pixel count in top9=0
```

Large-batch generator-init rerun of this baseline:

```text
artifact:
  results/topopt_best40_sorted_material_centered_l2_bs512_buf1024_500_seed0/top_designs_curiosity_0.0003_seed_0.npz

same current-best baseline, except:
  batch_size=512
  buffer_multiplier=2
  buffer_size=1024
  n_iter=500

500 iter seed 0:
  eval_count=257024
  best_feasible_compliance=73.8842
  mean top9 compliance=73.9672
  feasible_rate=1.0
  mean_hamming=0.0201
  intermediate-density pixel count in top9=0
  runtime=24:52
```

The large-batch run had a long delayed improvement phase: it stayed above
`220` until roughly iteration 160, then reached the 90s by iteration 220 and
the 70s by iteration 270. It did not beat the best seed-0 small-batch baseline
(`73.88` vs `72.52`), but it is competitive with the seed-2 small-batch result
and better than the blob-init 3k run.

Takeaway: full smoothing hurts the best-quality baseline. It quickly converges
to one nearly repeated grey-density bridge and does not approach the unsmoothed
baseline (`94.32` vs `72.52`). Blob init without smoothing is much healthier
and stays fully binary, but it still trails the generator-initialized seed-0
baseline (`76.44` vs `72.52`) and collapses to near-identical top designs. Use
`density_filter_radius=0` and generator init for the current best 40x20 setting
unless explicitly testing initialization effects.

Plain non-ranking GAN rerun at large batch:

```text
common setting:
  encoding=sorted_material
  sorted_material_profile=binary
  density_filter_radius=0
  projection_beta=0
  generator_type=conv
  generator_output_norm=centered_l2
  discriminator_type=mlp
  optimizer_type=default
  g_torch_optimizer=muon
  d_torch_optimizer=muon
  discriminator_steps=1
  batch_size=512
  buffer_multiplier=2
  buffer_size=1024
  n_iter=500
  seed=0

generator-init LR sweep:
  g=0.01, d=0.03: best_feasible=87.5324
  g=0.03, d=0.03: best_feasible=84.9266
  g=0.03, d=0.10: best_feasible=90.0508
  g=0.06, d=0.10: best_feasible=90.1839

blob-init LR sweep:
  g=0.01, d=0.03: best_feasible=90.8980
  g=0.03, d=0.03: best_feasible=100.6767
  g=0.03, d=0.10: best_feasible=120.5757
  g=0.06, d=0.10: best_feasible=220.6838
```

The large-batch plain GAN result is much better than the older sorted-binary
plain BCE GAN result (`134.7742` at 1k), but it remains behind the quantile
ranker baseline. Visually it learns a plausible but highly collapsed diagonal
cantilever family. Blob initialization is worse for plain GAN: it starts from a
better initial archive but trains more slowly and ends below generator init.

Plain GAN curiosity/diversity probe at the best generator-init LR
(`g_lr=0.03`, `d_lr=0.03`, 500 iterations):

```text
raw Wang-Isola uniformity, curiosity_reference=buffer:
  curiosity=0.001: best_feasible=86.8237
  curiosity=0.003: best_feasible=78.8834

Plummer repulsion in raw genome space:
  curiosity=0.001: best_feasible=78.9355
  curiosity=0.003: best_feasible=79.4305

boundary-Chamfer diversity as a buffer ladder objective:
  diversity:0.03: best_feasible=131.8867
  diversity:0.05: best_feasible=190.3852
```

For `curiosity_space=raw`, uniformity is applied to the centered-L2-normalized
raw generator score vectors, not to decoded binary topologies. With
`curiosity_reference=buffer`, the loss concatenates the generated batch with a
top-buffer batch, so gradients push current generated outputs away from each
other and away from elite buffer score vectors. The buffer samples are constants.
Plummer uses the same raw score vectors, embedded by the Plummer loss with
layernorm. Topology-space curiosity is not available for `sorted_material`, so
it was not tested here. Chamfer diversity is not a G-side curiosity loss; it is
a buffer objective/constraint, and the tested bounds diluted compliance pressure
badly.

Takeaway: for plain GAN, small raw-space repulsion is useful and nearly closes
the gap to the ranker baseline at 500 iterations (`78.88` vs `73.88` for the
large-batch ranker run). The best plain-GAN variant is now raw uniformity
`0.003`, essentially tied with Plummer `0.001`. Both still collapse around one
motif but preserve more variation than no-curiosity GAN. Chamfer-style diversity
should not be used as a default buffer objective in this setup.

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

## Matrix-Free Compliance Solver Probe

The current custom SciPy FEM backend is forward-only but not matrix-free: for
each design it assembles a reduced sparse stiffness matrix and calls
`scipy.sparse.linalg.spsolve`. Population parallelism is currently
`ThreadPoolExecutor.map(...)` over designs.

Added a comparison utility:

```text
examples/topology_optimization/compare_matrix_free_cg.py
```

It loads saved physical designs, evaluates direct sparse compliance, then
solves the same systems with a batched matrix-free elasticity operator and
Jacobi-preconditioned CG.

40x20 sanity check on the blob-init baseline artifact:

```text
top_k=3, e_min/e_max=1e-3, tol=1e-8, max_iter=1000
direct vs CG compliance relative error: <= 4.6e-13
CG iterations: 358 for all three designs
```

TOM 150x100 check on
`results/tom_cantilever_blobinit_nosmooth_lr_g0p03_d0p03_iter1000_seed0`:

```text
top_k=1, e_min/e_max=1e-6, tol=1e-6, max_iter=5000
direct compliance = 0.0115316536762
CG compliance     = 0.0115316536573
relative error    = 1.64e-9
CG iterations     = 2656
```

For the TOM top-4 batch with the default `e_min/e_max=1e-6`, the first two
designs converged by 5000 iterations but the other two still had residuals
around `5e-5`. Compliance was still close:

```text
top_k=4, tol=1e-6, max_iter=5000
relative compliance error range: 1.6e-9 .. 1.3e-5
CG iterations: 2656, 4344, 5000, 5000
```

A fixed 2000-iteration CG budget on the same TOM batch gave roughly
`3e-5 .. 1.4e-4` relative compliance error. A fixed 1000-iteration budget gave
roughly `0.6%` error. This confirms the matrix-free operator is numerically
compatible, but TOM's `1e-6` stiffness floor creates the expected CG iteration
problem. A `1e-3` stiffness floor changes the direct compliance for these
binary TOM designs by about `1%`, so using that floor is not a pure solver
implementation detail; it changes the benchmark objective slightly.

Integrated the matrix-free path into the 40x20 optimizer loop with:

```text
--compliance_solver matrix_free_cg
--matrix_free_cg_max_iter 1000
--matrix_free_cg_tol 1e-6
```

50-iteration smoke on the current best 40x20 setting:

```text
matrix-free artifact:
  results/topopt_best40_sorted_material_centered_l2_matrixfreecg_smoke50_seed0/top_designs_curiosity_0.0003_seed_0.npz

direct artifact:
  results/topopt_best40_sorted_material_centered_l2_direct_smoke50_seed0/top_designs_curiosity_0.0003_seed_0.npz

matrix-free final best_feasible_compliance = 194.2675
direct final best_feasible_compliance      = 194.2675
matrix-free mean top9 compliance           = 284.5922
direct mean top9 compliance                = 284.5922
saved top designs equal                    = true
actual compliance max abs diff             = 0.0
```

Runtime on local CPU:

```text
matrix-free 50 iter: 3:00 total, initial buffer 0:29
direct 50 iter:      0:26 total, initial buffer 0:02
```

MPS/Metal smoke on the same 50-iteration setting:

```text
artifact:
  results/topopt_best40_sorted_material_centered_l2_matrixfreecg_mps_smoke50_seed0/top_designs_curiosity_0.0003_seed_0.npz

--matrix_free_cg_device mps
--matrix_free_cg_dtype float32

MPS matrix-free final best_feasible_compliance = 208.1269
direct re-score of MPS top design               = 207.8801
MPS matrix-free mean top9 compliance            = 299.4609
runtime                                         = 3:02 total
initial buffer                                  = 0:31
```

The MPS path works only in float32. PyTorch MPS does not implement
`index_copy`, so the active-subset CG optimization is disabled on MPS and full
batch matvecs are used. On this 40x20 CPU-local benchmark, MPS gives no speedup
over CPU matrix-free CG and is much slower than direct sparse solves. The
float32 path also changes early ranking enough to select a worse top design
(`207.88` direct re-score vs `194.27` for CPU/direct at 50 iterations).

MPS large-batch throughput probe:

```text
artifact:
  results/topopt_best40_sorted_material_centered_l2_matrixfreecg_mps_batch512_smoke10_seed0/top_designs_curiosity_0.0003_seed_0.npz

--batch_size 512
--buffer_multiplier 1
--n_iter 10

initial buffer: 512 designs in one MPS batch, 0:06
full run: 5632 evaluations, 1:14 optimizer time
best_feasible_compliance: 452.4799
```

Compared with the earlier MPS `batch_size=64` smoke, the one-shot 512-design
evaluation is much better for GPU utilization (`0:06` for 512 vs `0:31` for
8x64 chunks). This is a throughput win per compliance. It is not a free training
win: at fixed evaluation budget, larger batches mean fewer G/D/ranker update
steps, and this 10-step batch-512 probe did not improve beyond the initial
buffer.

Takeaway: the matrix-free CG loop is optimizer-compatible and matches the
direct solver on the small problem, but the current PyTorch CPU implementation
is much slower than sparse direct solves at 40x20. The current PyTorch MPS path
is not better for this workload. This remains useful as a correctness prototype
for a future GPU/JAX batched version, not as a replacement for the local
small-grid CPU baseline.

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

## Multi-Niche Elite Buffers

Implemented an example-local `NicheEliteBuffer` in `cantilever_fem.py`. It
keeps several independent elite buffers, chooses niche representatives by
decoded-design Hamming distance, assigns candidates to nearest niches, and
exposes either a `balanced` round-robin view or a globally sorted view to D and
history. Relevant flags:

```text
--niche_buffer_count N
--niche_buffer_min_hamming H
--niche_buffer_view_mode balanced|global
```

First tests used the strong blob + Plummer baseline:

```text
grid=40x20, encoding=sorted_material, sorted_material_profile=binary
G=conv, G output norm=centered_l2, D=mlp, spectral norm=true
optimizer=quantile_ranked_default, tau=4, ranker_weight=1
Muon/Muon, g_lr=0.03, d_lr=0.1
batch_size=64, curiosity=0.003, curiosity_space=plummer
initial_buffer_mode=random_blobs
```

Results at 500 iterations:

```text
single buffer reference:
  buffer_multiplier=2
  best_feasible=83.186
  final 3k top9 mean_hamming=0.0279

4 niches, balanced view:
  buffer_multiplier=4, total buffer=256, hamming=0.2
  best_feasible=166.182
  mean top/archive=540.894
  top9 mean_hamming=0.1048
  conclusion: too much weak-niche retention, D/G signal diluted.

4 niches, global view:
  buffer_multiplier=4, total buffer=256, hamming=0.2
  ranker_sample_pool_size=256
  best_feasible=137.175
  mean top/archive=240.815
  top9 mean_hamming=0.0847
  conclusion: better than balanced but still over-constrained.

2 niches, balanced view:
  buffer_multiplier=2, total buffer=128, hamming=0.1
  best_feasible=81.500
  mean top/archive=97.303
  top9 mean_hamming=0.0705
  conclusion: promising; beats the 500-iter single-buffer trace while keeping
  more diversity.
```

Important implementation correction: the first niche rebuild underfilled the
archive when nearest-representative clusters were imbalanced. The current code
fills underfull niches from ranked overflow, so the buffer reaches full
capacity when enough candidates exist.

Longer 2-niche continuation:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same 2-niche balanced setting, 3k iterations:
  best_feasible=76.9707
  mean top/archive=78.1284
  mean top9 compliance=77.2283
  mean_hamming=0.0379
  feasible_rate=1.000

per-niche final state:
  niche 0: best=76.9707, mean=77.6678
  niche 1: best=77.9342, mean=78.5889
  representative min Hamming=0.0400
```

Takeaway: the 2-niche archive remains viable and reaches a good result, but it
does not beat the single-buffer blob+Plummer 3k reference (`74.0962`). The
representatives also drift far below the requested `0.1` Hamming separation by
the end. This confirms the next niche work should preserve identity more
directly rather than simply extending balanced 2-niche training.

Implementation update: the niche buffer no longer rebuilds all niches globally
on every insert. It now seeds stable niche anchors, routes each new candidate to
the nearest anchor by decoded-design Hamming distance, and inserts only into
that niche's private buffer. Existing samples do not move between niches. The
combined `balanced`/`global` views still expose samples from all niches to the
shared G/D training path.

First no-leak run:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same 2-niche balanced setting, stable-anchor/no-leak inserts, 3k iterations:
  best_feasible=99.6439
  mean top/archive=774.0004
  mean top9 compliance=100.0028
  mean_hamming=0.4315
  feasible_rate=1.000

per-niche final state:
  niche 0: best=99.6439, mean=100.6197
  niche 1: best=523.2489, mean=1447.3811
  representative min Hamming=0.5325
```

Takeaway: strict no-leak routing preserves niche identity, but it starves the
weak niche. The shared G discovers candidates near one anchor, while the other
niche remains essentially at its initial blob quality. The useful next variant
is not pure no-leak balanced exposure; it needs per-niche proposal pressure
(sample G around each anchor, condition G on niche id, or route generated
batches deliberately per niche) while still allowing D to compare across niches.

Fixed clustered input-latent follow-up:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_fixedz_clusters_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak 2-niche setting, plus:
  --fixed_latent_bank
  --fixed_latent_selection clustered_niches
  --fixed_latent_sample_mode balanced_niches
  --fixed_latent_bank_size 128
  --fixed_latent_niche_count 2
  --fixed_latent_niche_center_scale 4.0
  --fixed_latent_niche_within_std 0.35

latent bank geometry:
  within mean L2=3.4732
  between min L2=10.1566
  separation margin L2=5.0861

3k final:
  best_feasible=187.4819
  mean top/archive=658.3960
  mean top9 compliance=215.4255
  mean_hamming=0.3813
  feasible_rate=1.000

per-niche final state:
  niche 0: best=187.4819, mean=238.5990
  niche 1: best=523.2489, mean=1078.1930
  representative min Hamming=0.7000
```

Takeaway: fixed clustered z values do enforce latent-side separation, and they
make the weak niche less bad than strict no-leak alone (`mean 1447 -> 1078`).
But the good niche quality collapses (`best 99.64 -> 187.48`, `top9 100.00 ->
215.43`). This looks like a proposal-reach problem: the hard fixed bank and
balanced latent batches preserve identity, but they make the shared G less able
to exploit the one niche that was learning useful structures. If revisiting this
variant, add mild z jitter or use a condition/niche-id input with fresh samples
inside each latent cluster instead of a fully fixed bank.

Niche-local rank-target follow-up:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak 2-niche setting, no fixed latent bank, plus:
  --ranker_niche_local_targets

mechanism:
  D real loss is built from separate rank lists inside each niche.
  The best sample in each private niche receives the high quantile target for
  that niche, instead of weak-niche samples being assigned low targets because
  the other niche has globally lower compliance.

3k final:
  best_feasible=75.5635
  mean top/archive=76.5664
  mean top9 compliance=75.7787
  mean_hamming=0.0338
  feasible_rate=1.000

per-niche final state:
  niche 0: best=76.4415, mean=77.1168
  niche 1: best=75.5635, mean=76.0159
  representative min Hamming=0.0250
```

Takeaway: this directly fixes the starvation failure. Both niches now receive
useful D targets, and the final quality beats the earlier leaky 2-niche run
(`76.9707`) while avoiding the catastrophic weak-niche archive (`1447` mean).
However, it does not preserve meaningful niche identity: the final
representative distance collapses to `0.025`, below both the requested `0.1`
and the old leaky run's `0.04`. The next useful variant should keep
`--ranker_niche_local_targets`, then add an explicit representative-distance or
per-niche proposal-distance constraint so quality and identity are enforced at
the same time.

Next reasonable niche experiments:

```text
1. Keep niche-local D targets, then add a minimum final-representative-distance
   rule or per-niche proposal-distance filter before insertion.
2. Condition G/D on niche id only after the representative-distance constraint
   is tested; the local rank-target result shows conditioning is not required
   for quality.
3. Sweep hamming in {0.05, 0.1, 0.15} after quality+identity works together.
4. Try 3 niches only if 2 niches improves after identity preservation; 4 niches
   was too strong.
```

Follow-up implementation:

```text
--initial_blob_cluster_niches
```

This clusters the random blob seed candidates into balanced Hamming-distance
niches before initial buffer fill. The initial labels are queued into
`NicheEliteBuffer`, so the replayed/evaluated seed codes are inserted directly
into their assigned niche. Artifacts now save:

```text
niche_history
niche_history_columns
initial_blob_niche_labels
initial_blob_cluster_internal_mean_hamming
initial_blob_cluster_external_mean_hamming
initial_blob_cluster_separation_margin
```

First clustered-init test, same 2-niche 500-iter setting:

```text
init cluster internal mean Hamming=0.2322
init cluster external mean Hamming=0.6729
init cluster margin=0.4407

final best_feasible=87.4029
final top9 mean_hamming=0.0687

per-niche best at 500:
  niche 0: 87.4029, mean=90.486
  niche 1: 91.268,  mean=100.180
  representative min Hamming=0.0625
```

Takeaway: clustered init gives a cleaner and better-separated start, but in
this configuration it underperforms unclustered 2-niche init (`81.5005`) and the
niche representatives still collapse toward the same structural family. The
next useful variant is not stronger initial clustering; it is preserving niche
identity longer, e.g. by using stable medoids/anchors or a minimum
representative-distance constraint during rebuild.

Niche-output separation follow-up:

```text
new flags:
  --niche_output_separation_weight W
  --niche_output_separation_margin M

mechanism:
  During G updates, decode the generated batch into topology proxies, assign
  each generated sample to the nearest stable niche anchor, compute centered-L2
  raw-output prototypes for each represented niche, and penalize prototype
  distances below M. This is differentiable through raw G outputs, while the
  niche assignment is the same decoded-proximity rule used for insertion.
```

Smoke, 60 iterations:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_sep_w0p25_m035_smoke60_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker setting, plus:
  --niche_output_separation_weight 0.25
  --niche_output_separation_margin 0.35

final:
  best_feasible=329.9935
  top9 mean_hamming=0.4137

per-niche:
  niche 0: best=367.7154, mean=424.7095
  niche 1: best=329.9935, mean=459.3786
  representative min Hamming=0.5275
```

The smoke confirmed the loss was active: early generated families were very
different across niches, but quality was still far from meaningful.

500-iteration mild separation:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_sep_w0p25_m035_500_seed0/top_designs_curiosity_0.003_seed_0.npz

final:
  best_feasible=107.0810
  mean top/archive=112.1495
  mean top9 compliance=108.0687
  mean_hamming=0.0567
  feasible_rate=1.000

per-niche:
  niche 0: best=110.2407, mean=110.9729
  niche 1: best=107.0810, mean=107.6501
  representative min Hamming=0.06375
```

500-iteration stronger separation:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_sep_w1_m05_500_seed0/top_designs_curiosity_0.003_seed_0.npz

same setting, plus:
  --niche_output_separation_weight 1.0
  --niche_output_separation_margin 0.5

final:
  best_feasible=117.4045
  mean top/archive=123.7170
  mean top9 compliance=119.9908
  mean_hamming=0.0570
  feasible_rate=1.000

per-niche:
  niche 0: best=117.4045, mean=119.3847
  niche 1: best=120.6976, mean=121.5489
  representative min Hamming=0.03875
```

Takeaway: raw-output prototype separation is a partial/negative result. It
raises 500-iteration representative distance relative to the 3k local-ranker
collapse (`0.025 -> 0.06375` in the mild case), but it does not clear the
requested `0.1` Hamming separation and the designs visually remain the same
diagonal-bridge family. Increasing the loss weight makes quality worse and does
not improve final decoded representative distance. The next attempt should
enforce separation in decoded/topology space at insertion or selection time,
not only through a raw-output G-side prototype penalty.

Decoded cross-niche insertion gate:

```text
new flags:
  --niche_buffer_cross_min_hamming H
  --niche_buffer_cross_reference_top_k K

mechanism:
  For normal NicheEliteBuffer insertions, route each decoded topology proxy to
  the nearest stable niche anchor, then reject it if it is less than H Hamming
  distance from the top K reference designs in any other niche. Initial queued
  niche seeds are not filtered. D/G still see the shared balanced/global view;
  the constraint is applied only at insertion.
```

500-iteration gate-only run:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_500_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker setting, plus:
  --niche_buffer_cross_min_hamming 0.1
  --niche_buffer_cross_reference_top_k 1

final:
  best_feasible=108.2415
  mean top9 compliance=109.8086
  mean_hamming=0.1074
  feasible_rate=1.000

per-niche:
  niche 0: best=111.5796, mean=114.5568
  niche 1: best=108.2415, mean=109.2404
  representative min Hamming=0.1600
  cross-top min Hamming=0.0925
  cross-top mean Hamming=0.1429
```

3k gate-only continuation:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --n_iter 3000

final:
  best_feasible=94.1112
  mean top9 compliance=94.1866
  archive mean compliance=96.9367
  mean_hamming=0.1199
  feasible_rate=1.000

per-niche:
  niche 0: best=98.4199, mean=98.6250
  niche 1: best=94.1112, mean=94.1529
  representative min Hamming=0.1975
  cross-top min Hamming=0.16875
  cross-top mean Hamming=0.1897
```

Important caveat: the run above used the default density filter and did not set
`--hard_binarize`. Even with `--sorted_material_profile binary`, the binary
sorted-material mask is passed through `density_filter_radius=1`, so the saved
and evaluated physical densities contain intermediate material values at
boundaries.

3k hard-binary gate run:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_hardbin_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --n_iter 3000
  --hard_binarize

final:
  best_feasible=78.6955
  mean top9 compliance=79.0188
  archive mean compliance=79.7406
  mean_hamming=0.0828
  feasible_rate=1.000
  intermediate-density pixel count=0

per-niche:
  niche 0: best=79.6075, mean=79.7189
  niche 1: best=78.6955, mean=78.9646
  representative min Hamming=0.1000
  cross-top min Hamming=0.1000
  cross-top mean Hamming=0.1203
```

Takeaway: decoded-space insertion gating plus hard binarization gives the best
quality result so far that uses only full-material/void physical densities. It
also beats the grey-density 3k gate run on compliance (`94.1112 -> 78.6955`),
so the grey boundary pixels were not required for quality. The tradeoff is that
the niche identity is now weaker: the cross gate is binding exactly at the
`0.1` threshold, top-9 mean Hamming drops to `0.0828`, and visually both rows
are the same diagonal-bridge family with small edge/web-width differences. This
is the current best manufacturable-style baseline, while the grey-density 3k
run remains stronger evidence for larger visual niche separation.

Because the gate only compares each candidate against the current top K
other-niche references at insertion time, it is still not a general all-pairs
guarantee. The next stricter run should increase
`--niche_buffer_cross_reference_top_k` or add a final/selection-time all-pairs
cross-niche distance constraint.

Staged density-filter removal:

```text
--density_filter_warmup_iters N
--density_filter_final_radius 0
```

This keeps `--density_filter_radius` active for iterations `1..N`, then switches
the evaluator to the final radius before iteration `N+1`. The current archive is
re-evaluated and rebuilt under the new radius at the switch, so old smoothed
compliance values do not keep ranking the buffer. Artifacts now record
`density_filter_active_radius`, `density_filter_switched`, and
`density_filter_extra_eval_count`.

This should be the next version of the niche run if we want smoothing as an
early optimization aid but final designs/evaluation without smoothed material.

3k staged-filter run:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_filterwarm500_final0_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --density_filter_radius 1
  --density_filter_warmup_iters 500
  --density_filter_final_radius 0

final:
  best_feasible=144.9760
  mean top9 compliance=156.3957
  mean_hamming=0.3527
  feasible_rate=1.000
  intermediate-density pixel count=0
  density_filter_active_radius=0
  density_filter_extra_eval_count=24

per-niche:
  niche 0: best=144.9760, top6 mean=155.5201, full-buffer mean=163.1098
  niche 1: best=150.1051, top6 mean=170.3927, full-buffer mean=183.4227
  representative min Hamming=0.2975
```

Takeaway: staged smoothing removal did what it was supposed to do mechanically:
the final archive is all full material/void, and the switch re-scored the
buffer before continuing. It strongly preserves niche separation, much better
than the hard-binary gate run (`rep min Hamming 0.2975` vs `0.1000`), but
quality is much worse (`144.98` vs `78.70`). This looks like a diversity-heavy
variant rather than a new best-quality baseline. If continuing from here, try a
shorter smoothing warmup or a lower cross-Hamming threshold after the switch.

3k staged-filter run, later switch:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_filterwarm1500_final0_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --density_filter_radius 1
  --density_filter_warmup_iters 1500
  --density_filter_final_radius 0

final:
  best_feasible=129.8338
  mean top9 compliance=133.2384
  mean_hamming=0.2848
  feasible_rate=1.000
  intermediate-density pixel count=0
  density_filter_active_radius=0
  density_filter_extra_eval_count=24

per-niche:
  niche 0: best=129.8338, top6 mean=131.7557, full-buffer mean=136.5049
  niche 1: best=132.5668, top6 mean=145.9197, full-buffer mean=152.3696
  representative min Hamming=0.3350
```

Takeaway: switching later helps. Compared with the 500-switch run, compliance
improves (`144.98 -> 129.83`) while decoded niche separation remains strong
(`rep min Hamming 0.2975 -> 0.3350`). This suggests the poor quality of the
500-switch version was partly due to cutting over before the smoothed objective
had learned a decent topology. A 2000-switch run is worth checking next.

3k staged-filter run, 2000-step switch:

```text
artifact:
  results/topopt_blob_niche2_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_filterwarm2000_final0_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --density_filter_radius 1
  --density_filter_warmup_iters 2000
  --density_filter_final_radius 0

final:
  best_feasible=91.5336
  mean top9 compliance=95.3073
  mean_hamming=0.0853
  feasible_rate=1.000
  intermediate-density pixel count=0
  density_filter_active_radius=0
  density_filter_extra_eval_count=24

per-niche:
  niche 0: best=91.5336, top6 mean=96.4382, full-buffer mean=98.5198
  niche 1: best=93.7350, top6 mean=95.7280, full-buffer mean=97.1832
  representative min Hamming=0.1225
```

Takeaway: 2000-step smoothing is the best staged-filter setting so far. It is
fully binary after switch, beats the 500- and 1500-switch runs on compliance,
and still keeps representative separation above the target (`0.1225`). It is
also competitive with the grey-density 3k gate run on best compliance
(`91.53` vs `94.11`) while avoiding intermediate material. It still does not
match the hard-binarized filter baseline on quality (`78.70`), but that run had
weaker niche identity (`rep min Hamming 0.1000`). This is now the better
quality/diversity compromise among the no-smoothing-at-finish runs.

5-niche smoothed stress test:

```text
artifact:
  results/topopt_blob_niche5_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_smoothed_3k_seed0/top_designs_curiosity_0.003_seed_0.npz

same smoothed no-leak + niche-local-ranker + cross-Hamming gate setting, plus:
  --niche_buffer_count 5
  --density_filter_radius 1

final:
  best_feasible=178.3232
  mean top9 compliance=210.5140
  archive mean compliance=279.1735
  mean_hamming=0.3959
  feasible_rate=1.000
  intermediate-density pixel count in top9=2777

per-niche:
  niche 0: best=186.2429, mean=216.1273
  niche 1: best=240.9352, mean=281.3757
  niche 2: best=272.4778, mean=304.2282
  niche 3: best=349.1482, mean=373.2044
  niche 4: best=178.3232, mean=206.3714
  representative min Hamming range=0.1600..0.2900
```

Takeaway: five niches are too many for the current 24-slot archive and shared
generator pressure. The run preserves visual diversity and all niches remain
feasible, but several niches are weak, and the best result is far worse than
the 2-niche smoothed gate run (`178.32` vs `94.11`). If revisiting 5 niches,
increase buffer size and probably batch size before changing the optimizer.

5-niche smoothed 10k continuation:

```text
artifact:
  results/topopt_blob_niche5_h010_plummer_no_leak_niche_local_ranker_crossham010_top1_smoothed_10k_seed0/top_designs_curiosity_0.003_seed_0.npz

same 5-niche smoothed setting, plus:
  --n_iter 10000
  --history_interval 50

final:
  best_feasible=105.3669
  mean top9 compliance=107.5479
  archive mean compliance=119.8044
  mean_hamming=0.1433
  feasible_rate=1.000
  intermediate-density pixel count in top9=2199

per-niche:
  niche 0: best=105.3669, mean=106.9502
  niche 1: best=111.4819, mean=112.0167
  niche 2: best=138.1243, mean=139.8225
  niche 3: best=126.0363, mean=129.6359
  niche 4: best=107.4258, mean=108.2950
  representative min Hamming range=0.1000..0.1100
```

Takeaway: longer training materially improves the 5-niche setting
(`178.32 -> 105.37` best feasible), so the 3k result was undertrained. However,
the five niches mostly converge to variants of the same diagonal bridge; the
representative distances sit exactly at the cross-Hamming threshold, and two
niches remain notably weaker. This is not a compelling 5-niche win yet. The
next fair 5-niche attempt should scale capacity, e.g. increase buffer and batch
so each niche gets at least the same number of slots as the 2-niche run.
