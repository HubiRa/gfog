# Topology Optimization Experiment Handoff

This summarizes the topology-optimization work so the next agent can continue on a larger NVIDIA machine without re-discovering the setup.

## Current Best Direction

The strongest setup so far is:

- `encoding=topk_volume`
- `optimizer_type=lsgan`
- `G=ConvDecoderGenerator`
- `D=MLP`
- `G optimizer=Muon`
- `D optimizer=Muon`
- `g_lr=0.03`
- `d_lr=0.03`
- `curiosity=0` or scheduled topology curiosity
- `grid=40x20`
- `volume_max=0.48`
- `density_filter_radius=1`
- `projection_beta=1`

Best completed result:

```text
Conv G + MLP D, 10k iters, seed 0
best_feasible_compliance = 96.2709
mean_top9                = 96.5351
relative best            = 2.4532
runtime                  = 1:28:27 on local CPU
```

Artifact:

```text
results/fem_cantilever_topkvolume_lsgan_iter10000_convG_mlpD_gmuon0p03_dmuon0p03_seed0/top_designs_curiosity_0_seed_0.npz
```

Scheduled topology curiosity reached the same quality much faster:

```text
Conv G + MLP D, 3k iters, topology curiosity 0.1, warmup_cosine
best_feasible_compliance = 96.1404
mean_top9                = 96.2538
relative best            = 2.4499
runtime                  = 0:29:59 on local CPU
```

Artifact:

```text
results/fem_cantilever_topkvolume_lsgan_iter3000_convG_mlpD_topocuriosity0p1_sched_gmuon0p03_dmuon0p03_seed0/top_designs_curiosity_0.1_seed_0.npz
```

## Important Code Changes

Main file:

```text
examples/topology_optimization/cantilever_fem.py
```

Added:

- `soft_volume` encoding.
- Differentiable decoded-density training via `--train_on_decoded`.
- Corrected metrics: `archive_best_compliance`, `best_feasible_compliance`, `best_any_compliance`, feasibility counts/rates.
- Local experimental `Muon` optimizer selectable via `--g_torch_optimizer muon` and `--d_torch_optimizer muon`.
- `TopologySpaceUniformity`, applying Wang-Isola uniformity to decoded topology fields.
- `--curiosity_space raw|topology`.
- `--curiosity_schedule none|warmup_cosine`.
- `ConvDecoderGenerator`, enabled with `--generator_type conv`.
- `ConvDiscriminator`, enabled with `--discriminator_type conv`.
- Network metadata saved into `.npz` artifacts.

Also added:

```text
examples/topology_optimization/run_overnight_topopt.sh
```

## Key Experimental Results

Early baseline evolution:

```text
MLP G + MLP D, Adam/Adam, topk_volume, 1000 iters:       290.9863
MLP G + MLP D, Muon/Adam, topk_volume, 1000 iters:       232.4330
MLP G + MLP D, Muon/SGD, topk_volume, 1000 iters:        229.2893
MLP G + MLP D, Muon/Muon, topk_volume, 1000 iters:       214.0132
MLP G + MLP D, Muon/Muon, topk_volume, 2000 iters:       180.2031
MLP G + MLP D, Muon/Muon, topk_volume, 10000 iters:      171.5710
Conv G + MLP D, Muon/Muon, topk_volume, 3000 iters:      100.5034
Conv G + MLP D, Muon/Muon, topk_volume, 10000 iters:      96.2709
Conv G + MLP D, scheduled topology curiosity, 3000 iters: 96.1404
```

Larger batch and larger G before Conv G:

```text
MLP G + MLP D, 1000 iters, batch128:       195.4948
MLP G + MLP D, 3000 iters, batch128:       179.7507
MLP G + MLP D, 3000 iters, seed1:          189.5316
MLP G + MLP D, 3000 iters, G=[256,256,256]:184.1259
```

Conv discriminator was not good in the first test:

```text
Conv G + Conv D, 1000 iters: 356.8725
```

Do not prioritize Conv D as implemented unless changing architecture/training.

## Curiosity Findings

Raw-space curiosity on `topk_volume` increases diversity but hurts compliance.

```text
topk_volume + LSGAN + MLP G/D + 1000 iters:
curiosity=0 raw: best 290.9863, hamming 0.1196
curiosity=5 raw: best 327.3830, hamming 0.4401
```

Topology-space curiosity is more meaningful but fixed weights still hurt on the MLP setup:

```text
topology curiosity=0.1 fixed: best 309.0334
topology curiosity=1 fixed:   best 403.0681
```

Scheduled topology curiosity is promising with Conv G:

```text
Conv G + MLP D, 3k:
no curiosity:                         best 100.5034
topology curiosity=0.1 warmup_cosine: best  96.1404
```

## Commands To Reproduce Best Runs

Best no-curiosity Conv G 10k:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 --grid_height 20 \
  --n_iter 10000 --batch_size 64 --buffer_multiplier 2 \
  --latent_dim 64 \
  --encoding topk_volume \
  --optimizer_type lsgan \
  --curiosity 0 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter10000_convG_mlpD_gmuon0p03_dmuon0p03_seed0
```

Best scheduled-curiosity Conv G 3k:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 --grid_height 20 \
  --n_iter 3000 --batch_size 64 --buffer_multiplier 2 \
  --latent_dim 64 \
  --encoding topk_volume \
  --optimizer_type lsgan \
  --curiosity 0.1 \
  --curiosity_space topology \
  --curiosity_schedule warmup_cosine \
  --curiosity_warmup_frac 0.05 \
  --curiosity_min 0 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter3000_convG_mlpD_topocuriosity0p1_sched_gmuon0p03_dmuon0p03_seed0
```

## Recommended Large-Scale Experiments

Prioritize these on the NVIDIA machine:

1. Seed sweep for the best scheduled-curiosity Conv G config:

```text
seeds: 0, 1, 2, 3, 4
n_iter: 3000 and 10000
```

2. Longer scheduled-curiosity Conv G:

```text
n_iter: 10000, 20000
curiosity: 0.05, 0.1, 0.2
curiosity_schedule: warmup_cosine
warmup_frac: 0.02, 0.05, 0.1
```

3. Conv G capacity:

```text
generator_channels: 64, 96, 128
latent_dim: 64, 128
```

4. Batch/buffer:

```text
batch_size: 64, 128, 256
buffer_multiplier: 2, 4
```

5. Geometry/mesh scale once robust:

```text
grid: 60x30, 80x40
```

Keep `D=MLP` initially. The first Conv D test was poor and slow.

## Caveats

- Results are single-seed unless noted; seed sweep is required before claiming robustness.
- Current FEM is CPU SciPy sparse solve; GPU only helps the neural nets unless a GPU FEM backend is integrated.
- `topk_volume` is discrete. The discriminator sees raw scores in the best setup; the evaluator applies top-k projection.
- The local Muon implementation is experimental and self-contained in the example file.
- Do not compare `archive_best_compliance` blindly across different value-level layouts; use `best_feasible_compliance`.
