# Topology Optimization Handover, June 2026

This handover captures the state of the 40x20 and TOM topology optimization
work before moving to a larger GPU machine such as NVIDIA Spark.

## Current Branch

```text
branch: research/topopt
```

Generated result artifacts live under `results/` and are intentionally not part
of the repo commit.

## Current Best 40x20 Setting

Use this as the reference before changing anything else:

```text
encoding=sorted_material
sorted_material_profile=binary
density_filter_radius=0
projection_beta=0
generator_type=conv
generator_output_norm=centered_l2
discriminator_type=mlp
optimizer_type=quantile_ranked_default
ranker_list_size=64
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

Best recorded 40x20 results:

```text
3k seed 0: best_feasible_compliance=72.5222
3k seed 1: best_feasible_compliance=78.1218
3k seed 2: best_feasible_compliance=74.2828
```

The strongest large-batch ranker probe used the same setting except
`batch_size=512`, `buffer_multiplier=2`, and `n_iter=500`:

```text
best_feasible_compliance=73.8842
mean top9 compliance=73.9672
eval_count=257024
```

## Ranking Objective

The current best optimizer is not plain GAN. It is
`quantile_ranked_default`, which uses a discriminator as a dense rank target
model over the elite buffer.

For each ranker step:

1. Sample a sorted list of real elite buffer entries.
2. Train `D(real)` with BCE targets based on sorted rank.
3. Train `D(fake)` toward zero for generated samples.
4. Train `G` to make generated samples score like real/top samples.

For the current best setting, real rank targets are local-list exponential
targets:

```text
target(rank_index i) = exp(-i / tau)
tau = 4
```

With `ranker_sample_mode=random_top_pool` and `ranker_sample_pool_size=128`,
the ranker list is sampled from the current top tail instead of always using
the exact top-k list. This stochastic top-tail sampling mattered; exact top-k
ranker lists were worse in previous tests.

Important distinction:

```text
plain GAN:
  D(real elite)=1, D(fake generated)=0

quantile_ranked_default:
  D(real elite at rank i)=exp(-i/tau)
  D(fake generated)=0
  G tries to make generated samples score like real/top samples
```

The ranker is not a compliance regressor. It only sees ordered elite samples
and fake samples. True compliance remains a black-box FEM evaluation used to
insert generated samples into the elite buffer.

## Plain GAN Findings

Plain non-ranking GAN improved a lot when rerun with `batch_size=512` and
`buffer_size=1024`, but it still trails the ranker.

```text
plain GAN, generator init, 500 iter:
  g=0.01 d=0.03: best=87.5324
  g=0.03 d=0.03: best=84.9266
  g=0.03 d=0.10: best=90.0508
  g=0.06 d=0.10: best=90.1839

plain GAN, blob init, 500 iter:
  g=0.01 d=0.03: best=90.8980
  g=0.03 d=0.03: best=100.6767
  g=0.03 d=0.10: best=120.5757
  g=0.06 d=0.10: best=220.6838
```

Blob init is not useful for plain GAN. It starts from a better archive but
trains more slowly and ends worse than generator init.

Raw/Plummer repulsion made plain GAN much stronger:

```text
plain GAN, g=0.03 d=0.03, batch_size=512, buffer_size=1024, 500 iter:
  raw uniformity 0.001: best=86.8237
  raw uniformity 0.003: best=78.8834
  Plummer 0.001:        best=78.9355
  Plummer 0.003:        best=79.4305
  Chamfer ladder 0.03:  best=131.8867
  Chamfer ladder 0.05:  best=190.3852
```

For `curiosity_space=raw`, the repulsion is applied to the centered-L2 raw
generator score vectors, not decoded binary topologies. With
`curiosity_reference=buffer`, generated vectors are repelled from each other
and from top-buffer raw score vectors. The buffer vectors are constants; only
the generated vectors receive gradients.

Topology-space curiosity does not currently support `encoding=sorted_material`.
Boundary-Chamfer diversity is a buffer objective/constraint, not a G-side
curiosity loss. In these tests it diluted compliance pressure badly.

## Matrix-Free FEM State

The repo now has a matrix-free compliance solver path in
`examples/topology_optimization/cantilever_fem.py`:

```text
--compliance_solver matrix_free_cg
--matrix_free_cg_max_iter N
--matrix_free_cg_tol T
--matrix_free_cg_device cpu|mps|cuda
--matrix_free_cg_dtype float32|float64
```

There is also a comparison utility:

```text
examples/topology_optimization/compare_matrix_free_cg.py
```

Local validation:

```text
40x20 top designs:
  direct vs matrix-free CG relative error <= 4.6e-13

TOM 150x100, e_min/e_max=1e-6:
  top1 direct compliance = 0.0115316536762
  top1 CG compliance     = 0.0115316536573
  relative error         = 1.64e-9
  CG iterations          = 2656
```

CPU and MPS were slower than direct sparse solves for 40x20. That does not
invalidate the CUDA/Spark direction; it only says local 40x20 is too small and
direct SciPy is very efficient there. MPS requires `float32` and currently uses
a full-batch matvec fallback because PyTorch MPS lacks the active-subset
`index_copy` path used on CPU.

## Spark Experiment Queue

Run these in order on NVIDIA Spark or another CUDA machine.

1. Validate CUDA matrix-free ranking, not only compliance.

```bash
uv run python examples/topology_optimization/compare_matrix_free_cg.py \
  --artifact results/tom_cantilever_blobinit_nosmooth_lr_g0p03_d0p03_iter1000_seed0/top_designs_curiosity_0.0003_seed_0.npz \
  --top_k 8 \
  --device cuda \
  --dtype float32 \
  --max_iter 2000 \
  --tol 1e-6
```

Repeat with `--dtype float64` if the Spark setup has usable float64 throughput,
and with `--e_min_ratio 1e-3` to quantify the conditioning/objective tradeoff.

2. Batch-size scaling at fixed eval budget for the ranker.

Test:

```text
bs=64,   buffer=512
bs=256,  buffer=1024
bs=512,  buffer=1024
bs=1024, buffer=2048
```

Keep total evaluations comparable. The question is whether huge batches help
throughput enough to compensate for fewer G/D/ranker update steps.

3. Large TOM ranker with CUDA matrix-free FEM.

Start from:

```text
preset=tom_cantilever_2d
encoding=sorted_material
sorted_material_profile=binary
optimizer_type=quantile_ranked_default
ranker_target_curve=exp
ranker_tau=4
ranker_sample_pool_size=128
curiosity=0.0003
curiosity_space=raw
g_lr=0.03
d_lr=0.1
batch_size>=512
compliance_solver=matrix_free_cg
matrix_free_cg_device=cuda
```

Compare:

```text
generator init vs blob init
no smoothing
smoothing for 1500 iterations, then density_filter_radius=0
smoothing for 2000 iterations, then density_filter_radius=0
```

4. Treat plain GAN plus repulsion as a real branch.

Run longer and multi-seed:

```text
optimizer_type=default
g_lr=0.03
d_lr=0.03
batch_size=512 or 1024
raw uniformity: 0.001, 0.003, 0.006
Plummer: 0.0005, 0.001, 0.003
n_iter=2000 or 3000
seeds=0,1,2
```

This is worth doing because raw `0.003` reached `78.8834` in only 500
iterations, close to the ranker range.

## Avoid First

Do not spend first Spark time on:

```text
full smoothing from start to finish
blob init for plain GAN
Chamfer diversity ladder as the primary diversity mechanism
MPS performance extrapolations
```

## Useful Scripts

```text
examples/topology_optimization/run_topopt_plain_gan_bs512_lr_sweep.sh
examples/topology_optimization/run_topopt_plain_gan_bs512_curiosity_sweep.sh
examples/topology_optimization/run_tom_blobinit_lr_sweep.sh
examples/topology_optimization/compare_matrix_free_cg.py
```

The generated artifacts and images are under `results/`; keep those out of git.
