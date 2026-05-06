# Topology Optimization Experiment Handoff

This summarizes the topology-optimization work so the next agent can continue on a larger NVIDIA machine without re-discovering the setup.

## Current Best Direction

The strongest setup so far is now:

- `encoding=topk_volume`
- `optimizer_type=quantile_ranked_default`
- `ranker_target_curve=exp`
- `ranker_tau=4`
- `ranker_weight=1.0`
- `ranker_sample_pool_size=128`
- `G=ConvDecoderGenerator`
- `D=MLP`
- `G optimizer=Muon`
- `D optimizer=Muon`
- `g_lr=0.03`
- `d_lr=0.03`
- `curiosity=0`
- `grid=40x20`
- `volume_max=0.48`
- `density_filter_radius=1`
- `projection_beta=1`

Best completed result:

```text
Quantile-ranked vanilla GAN, exp tau=4, buffer=512, sample pool=128, 10k iters, seed 0
best_feasible_compliance = 90.7142
mean_top9                = 90.7837
relative best            = 2.3116
runtime                  = 1:20:06 on local CPU
```

Artifact:

```text
results/fem_cantilever_topkvolume_quantile_ranked_default_exp4_w1_iter10000_convG_mlpD_bufx8_pool128_seed0/top_designs_curiosity_0_seed_0.npz
```

Previous strongest LSGAN results:

```text
Conv G + MLP D, 10k iters, seed 0
best_feasible_compliance = 96.2709
mean_top9                = 96.5351
relative best            = 2.4532
runtime                  = 1:28:27 on local CPU

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

Recommended next Spark baseline:

```text
Conv G + MLP D, quantile-ranked vanilla GAN, exp tau=4, top-tail ranker pool, Muon/Muon
n_iter: 10000 or 20000
batch_size: 1024
buffer_multiplier: 16
ranker_sample_pool_size: 2048 initially
effective elite buffer size: 16384
```

This keeps the current best learning setup and spends the larger machine budget on a much larger elite buffer without making each iteration too expensive. The 40x20 topology tensors are small; the likely limiter is FEM evaluation wall-clock, not buffer memory. Increase `BATCH_SIZE` to `2048` or `4096` only after measuring iteration time.

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
- `--curiosity_reference buffer|batch`.
- `--curiosity_schedule none|warmup_cosine`.
- `--elite_sampling random_top_k|top_k`.
- `--elite_pool_size`.
- Official `Levels.ladder(...)` support via `--levels_ladder`.
- `--levels_ladder_final_open`.
- `ConvDecoderGenerator`, enabled with `--generator_type conv`.
- `ConvDiscriminator`, enabled with `--discriminator_type conv`.
- Optional generic image decoder prior via `--encoding tiny_decoder`.
  This uses `diffusers.AutoencoderTiny` with `--tiny_decoder_model madebyollin/taesd` by default and projects the decoded grayscale image to the target top-k volume.
- TOM/GiNN-style cantilever preset via `--preset tom_cantilever_2d`.
  This sets a 150x100 grid on a 1.5x1 domain, `E=196`, `volume_max=0.48`, no density filter, and two right-edge traction patches matching the local TOM config.
- CPU-parallel SciPy FEM batch evaluation via `--fem_workers N`.
  This parallelizes independent compliance solves across worker threads while preserving batch order. It is supported for `--backend scipy`; keep `fem_workers=1` for `torchfem`.
- Plackett-Luce/listwise ranker experiments:
  - `--optimizer_type plackett_luce`
  - `--optimizer_type buffer_plackett_luce`
  - `--optimizer_type ranked_lsgan`
  - `--optimizer_type ranked_default`
  - `--optimizer_type ranked_wgan`
  - `--optimizer_type quantile_ranked_default`
  - `--ranker_list_size`
  - `--ranker_steps`
  - `--ranker_weight`
  - `--ranker_target_curve linear|exp`
  - `--ranker_tau`
  - `--ranker_target_scope local|global`
  - `--ranker_sample_pool_size`
  - `--ranker_sample_mode random_top_pool|top_k`
- Network metadata saved into `.npz` artifacts.

Also added:

```text
examples/topology_optimization/run_overnight_topopt.sh
examples/topology_optimization/run_topopt_best_local.sh
examples/topology_optimization/run_topopt_ranker_sweep.sh
examples/topology_optimization/run_spark_best_topopt.sh
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
Conv G + MLP D, LSGAN, exact top-k elite, 3000 iters:      99.4382
Conv G + MLP D, Muon/Muon, topk_volume, 10000 iters:      96.2709
Conv G + MLP D, scheduled topology curiosity, 3000 iters: 96.1404
Conv G + MLP D, scheduled curiosity + exact top-k elite:   99.9234
```

Ranking/ranker experiments:

```text
Archive Plackett-Luce, naive G=-D(G), 1000 iters:                    447.6559
Archive Plackett-Luce, elite-margin G, 1000 iters:                   118.4872
Archive Plackett-Luce, elite-margin G, 3000 iters:                   111.0036
Buffer-only Plackett-Luce, ranker_steps=3, 1000 iters:               388.2138
Buffer-only Plackett-Luce, ranker_steps=1, d_lr=0.1, 1000 iters:     377.8967
Ranked LSGAN, implicit ranker_weight=1.0, 1000 iters:                183.1135
Ranked LSGAN, implicit ranker_weight=1.0, 3000 iters:                183.1135
Ranked LSGAN + topology curiosity, 1000 iters:                       448.9517
Ranked LSGAN, auxiliary ranker_weight=0.1, 1000 iters:               193.9727
Ranked vanilla GAN, auxiliary ranker_weight=0.1, 1000 iters:         167.2213
Ranked WGAN, auxiliary ranker_weight=0.1, 1000 iters:                406.6346
Quantile-ranked vanilla GAN, linear targets, w=1.0, 1000 iters:      227.4550
Quantile-ranked vanilla GAN, exp tau=4, w=1.0, 1000 iters:           183.4090
Quantile-ranked vanilla GAN, exp tau=8, w=1.0, 1000 iters:           100.5033
Quantile-ranked vanilla GAN, exp tau=8, w=1.0, 3000 iters:            92.1868
Quantile-ranked vanilla GAN, exp tau=8, w=1.0, topology curiosity:   267.4112
Quantile exp tau=8, 3k, topology curiosity 0.005:                    98.7260
Quantile exp tau=8, 3k, topology curiosity 0.01:                     93.0111
Quantile exp tau=8, 3k, topology curiosity 0.03:                    105.7145
Quantile exp tau=8, 3k, topology curiosity 1:                        92.2906
Quantile exp tau=8, 3k, topology curiosity 10:                      446.1272
Quantile exp tau=8, 3k, topology curiosity 100:                     309.8560
Quantile exp tau=4, 3k, batch-only topology curiosity 0.001:         168.5401
Quantile exp tau=4, 3k, batch-only topology curiosity 0.01:          177.8041
Quantile exp tau=8, 3k, buffer_multiplier=4, buffer=256:            104.1730
Quantile exp tau=8, 3k, buffer_multiplier=8, buffer=512:            109.4939
Quantile exp tau=8, 3k, buffer=512, sample_pool=128:                 94.2259
Quantile exp tau=4, 3k, buffer=512, sample_pool=128:                 91.3747
Quantile exp tau=4, 10k, buffer=512, sample_pool=128:                90.7142
Quantile exp tau=4, 3k, buffer=512, exact top_k=64 ranker list:      106.8030
Quantile exp tau=4, 3k, buffer=512, sample_pool=64, list=64:         106.8030
Quantile exp tau=4, 3k, buffer=512, sample_pool=256, list=64:         96.3954
Quantile exp tau=4, 3k, buffer=512, sample_pool=128, list=128:       225.4730
Quantile global exp tau=16, 3k, buffer=512, pool=128, list=64:       101.9539
Quantile global exp tau=32, 3k, buffer=512, pool=128, list=64:        95.4780
Quantile-ranked vanilla GAN, exp tau=16, w=1.0, 1000 iters:          111.2437
Quantile-ranked vanilla GAN, exp tau=32, w=1.0, 1000 iters:          114.8630
Tiny decoder TAESD prior, quantile exp tau=4, 1000 iters:            139.5329
Tiny decoder TAESD prior, LSGAN, 1000 iters:                         167.2432
```

Interpretation: listwise ranking is interesting conceptually, but the current PL implementations are not competitive with LSGAN. The best PL ranker result was archive Plackett-Luce with elite-margin G at `111.0036`, still behind Conv G LSGAN. Among the naive fake-rejection + PL-buffer variants, vanilla BCE GAN was best at `167.2213`, LSGAN was worse at `193.9727`, and WGAN was poor at `406.6346`. Curiosity made ranked LSGAN much worse by increasing diversity without usable compliance pressure.

The first non-PL dense-rank result is now the best overall direction: quantile-ranked vanilla GAN with exponential targets `exp(-rank / tau)` is very sensitive to `tau`. `tau=8` reached `100.5033` at 1k, matching the no-curiosity Conv G LSGAN 3k result, and improved to `92.1868` at 3k. Linear rank targets were poor at `227.4550`, `tau=4` was too sharp at `183.4090`, and softer `tau=16/32` were good but worse than `tau=8`.

Topology curiosity does not beat no-curiosity for quantile ranking in the 3k seed-0 sweep. `curiosity=1` nearly tied (`92.2906`) and had slightly higher diversity, but no curiosity remains best (`92.1868`). Small values were mixed: `0.01` nearly tied but collapsed harder, `0.005`/`0.03` were worse. Large values `10` and `100` are bad. Keep curiosity off for quantile ranking unless testing a different anti-collapse mechanism or much narrower schedules.

Batch-only topology curiosity was also bad for the current best quantile setting. `--curiosity_reference batch` avoids repelling from the elite buffer, but even `curiosity=0.001` degraded to `168.5401` and `0.01` degraded to `177.8041`. The Wang-Isola topology-uniformity term is therefore not the right anti-collapse mechanism for this setup, even batch-only.

Longer current-best run crossed below the previous 91-ish bound: 10k iterations reached `90.7142` with mean top-9 `90.7837`. It improved only modestly from 3k (`91.3747`) and collapsed more (`mean_hamming=0.0066`), but it is the current best compliance result.

Larger elite buffers did not help when ranker lists were sampled from the whole buffer. With batch `64`, buffer multiplier `2` means a buffer of `128` and reached `92.1868`. Buffer `256` degraded to `104.1730`; buffer `512` degraded to `109.4939`. The reason was likely rank-signal dilution. Adding `--ranker_sample_pool_size` fixed most of this: buffer `512` with ranker samples restricted to the top `128` reached `94.2259` at `tau=8`, and retuning to sharper `tau=4` reached the new best `91.3747`. Exact top-k ranker lists were worse (`106.8030`), so stochastic sorted sampling from the top-tail pool seems important. Pool-size/list-size ablations support this: pool `64`/list `64` was the same failure as exact top-k, pool `256`/list `64` was decent but worse (`96.3954`), and list `128` was very poor (`225.4730`). Global-rank targets were conceptually cleaner but worse: global `tau=16` got `101.9539`, global `tau=32` got `95.4780`. Keep local list ranks, list `64`, pool `128` for now.

## Ranking-GAN Design Space

Classic GAN structure:

```text
D(real) high
D(fake) low
G tries to make D(fake) high
```

For topology optimization, `real` means elite buffer samples and `fake` means newly generated samples. Because the buffer is ordered by true objective values, `D` can learn more than a binary real/fake decision.

Candidate realizations:

```text
Binary elite GAN:
  D(best buffer) -> 1, D(generated) -> 0, G pushes D(G) -> 1.

Weighted elite GAN:
  D(buffer_i) predicts a graded target from rank/quality, fake -> 0.

Ranked real + fake GAN:
  D ranks buffer samples correctly and rejects fake samples; G pushes fake samples high.

Bradley-Terry pairwise reward model:
  D(x_better) > D(x_worse) for sampled buffer pairs; G maximizes D(G).

Generated-vs-elite margin model:
  D(elite) should beat D(fake) by a margin; G tries to beat elite scores.

Quantile-ranked GAN:
  D(buffer_i) predicts rank quantile or exp(-rank/tau); D(fake) -> 0; G -> top target.

Energy-based version:
  D is an energy/fitness surrogate; good designs have low energy and G minimizes energy.
```

Most pragmatic next direction: quantile-ranked GAN. It keeps the stable fake/real pressure, gives dense per-buffer targets, avoids Plackett-Luce list normalization, and makes the top tail emphasis tunable through a target curve like `exp(-rank / tau)`.

Current naive ranker variants available:

```text
ranked_default: BCE fake rejection + PL(buffer ranking), G uses BCE real target
ranked_lsgan:   LSGAN fake rejection + PL(buffer ranking), G uses LSGAN real target
ranked_wgan:    WGAN fake pressure + PL(buffer ranking), G maximizes critic
```

Run the 1k naive ranker comparison with:

```bash
bash examples/topology_optimization/run_topopt_ranker_sweep.sh
```

## Official Ladder Mechanism

The topology example now supports the repo-native ladder mechanism:

```bash
--levels_ladder volume:0.55,0.50 compliance:130,110,100,95
```

This uses `Levels.ladder(...)` and `Rung.minimize(...)` directly. The evaluator returns raw objective values only for the objectives listed in `--levels_ladder`; the buffer expands those through the official ladder transform. This is separate from the older topology-specific args:

```text
--volume_ladder
--compliance_ladder
--roughness_ladder
--ladder_sequence
```

Do not combine the two mechanisms in one run.

The intended use is staged/alternating optimization. With `volume` and `compliance`, `Levels.ladder` interleaves rungs by default:

```text
volume#1, compliance#1, volume#2, compliance#2, ...
```

Recommended first official-ladder experiment with the current best ranker:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 --grid_height 20 \
  --n_iter 3000 --batch_size 64 --buffer_multiplier 8 \
  --latent_dim 64 \
  --encoding topk_volume \
  --optimizer_type quantile_ranked_default \
  --ranker_list_size 64 \
  --ranker_steps 1 \
  --ranker_sample_pool_size 128 \
  --ranker_sample_mode random_top_pool \
  --ranker_weight 1.0 \
  --ranker_target_curve exp \
  --ranker_target_scope local \
  --ranker_tau 4 \
  --curiosity 0 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --fem_workers 8 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --levels_ladder volume:0.55,0.50,0.48 compliance:130,110,100,95 \
  --output_dir results/fem_cantilever_topkvolume_quantile_ranked_default_levels_ladder_vol_comp_seed0
```

First result:

```text
Official Levels.ladder volume/compliance, 3k, seed 0
best_feasible_compliance = 96.4225
mean_top9                = 97.0245
```

This is worse than the current best `91.3747`. The transformed ladder values show that top designs already pass the volume and loose compliance gates; sorting is dominated by the final `compliance <= 95` violation and then raw compliance. So this first ladder acted mostly like a softened compliance objective near 95, not as a strongly staged material/compliance curriculum.

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
no curiosity, random elite batch:       best 100.5034
no curiosity, exact top-k elite batch:  best  99.4382
topology curiosity, random elite batch: best  96.1404
topology curiosity, exact top-k elite:  best  99.9234
quantile exp tau=4, top-tail sampling:  best  91.3747
```

## Commands To Reproduce Best Runs

TOM/GiNN-style cantilever setting:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --preset tom_cantilever_2d \
  --n_iter 3000 --batch_size 64 --buffer_multiplier 8 \
  --latent_dim 64 \
  --encoding topk_volume \
  --optimizer_type quantile_ranked_default \
  --ranker_list_size 64 \
  --ranker_steps 1 \
  --ranker_sample_pool_size 128 \
  --ranker_sample_mode random_top_pool \
  --ranker_weight 1.0 \
  --ranker_target_curve exp \
  --ranker_target_scope local \
  --ranker_tau 4 \
  --curiosity 0 \
  --projection_beta 1 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_tom_preset_quantile_tau4_seed0
```

The TOM preset solid-compliance scale is:

```text
solid_compliance = 0.0038145905
total vertical load = -0.2
```

This makes TOM-reported numbers such as `0.0059` plausible in the replicated units. It should not be compared directly to the old GFog toy scale where `solid_compliance=39.2425`.

Local timing check for 16 independent TOM-preset compliance evaluations:

```text
workers=1 time=5.164s
workers=2 time=2.708s
workers=4 time=1.347s
workers=8 time=0.805s
```

Use `--fem_workers 8` as the local default starting point and retune on the Spark. If memory pressure appears, reduce workers before reducing batch size.

First full local TOM-preset result with current best GFog setting:

```text
topk_volume + quantile_ranked_default, tau=4, 3k, seed 0, fem_workers=8
best_feasible_compliance = 0.0447193
mean_top9                = 0.0617
relative best            = 11.7232
solid_compliance         = 0.0038145905
mean_hamming             = 0.2500
runtime                  = 3:29:53
artifact                 = results/fem_cantilever_tom_preset_quantile_tau4_iter3000_workers8_seed0/top_designs_curiosity_0_seed_0.npz
```

This is now in the same absolute compliance unit scale as TOM/GiNN, but it is still much worse than the `~0.0059` target. The gap is not just unit scaling anymore; the current black-box GAN/ranker setup is underperforming on the TOM-resolution benchmark.

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
large-machine batch_size: 1024, 2048, 4096
buffer_multiplier: 2, 4, 8
Spark first shot: batch_size=1024, buffer_multiplier=16, buffer size=16384
```

5. Geometry/mesh scale once robust:

```text
grid: 60x30, 80x40
```

Keep `D=MLP` initially. The first Conv D test was poor and slow.

6. Tiny image decoder prior:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 --grid_height 20 \
  --n_iter 1000 --batch_size 64 --buffer_multiplier 8 \
  --latent_dim 64 \
  --encoding tiny_decoder \
  --tiny_decoder_model madebyollin/taesd \
  --tiny_decoder_latent_channels 4 \
  --tiny_decoder_latent_height 8 \
  --tiny_decoder_latent_width 8 \
  --tiny_decoder_latent_scale 1 \
  --optimizer_type quantile_ranked_default \
  --ranker_list_size 64 \
  --ranker_steps 1 \
  --ranker_sample_pool_size 128 \
  --ranker_sample_mode random_top_pool \
  --ranker_weight 1.0 \
  --ranker_target_curve exp \
  --ranker_target_scope local \
  --ranker_tau 4 \
  --curiosity 0 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type mlp \
  --generator_hidden_dims 256 256 \
  --discriminator_type mlp \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_tiny_decoder_quantile_tau4_seed0
```

This path is intentionally optional: local `diffusers` is not required for normal runs. Install `diffusers` and ensure the Hugging Face model is cached or downloadable before launching it on the Spark.

First local result:

```text
TAESD tiny decoder, 1k, seed 0
best_feasible_compliance = 139.5329
mean_top9                = 146.3702
mean_hamming             = 0.0874
runtime                  = 20:44
artifact                 = results/fem_cantilever_tiny_decoder_quantile_tau4_seed0/top_designs_curiosity_0_seed_0.npz
```

This is worse than the current `topk_volume` quantile-ranked baseline, but it is not an immediate collapse. The useful signal is diversity: hamming `0.0874` is much higher than the current best 10k run (`0.0066`). If continuing this path, try larger latent grids/scales or optimize the decoder latent directly with a different optimizer before spending 10k iterations.

LSGAN with the same tiny-decoder encoding was worse but more diverse:

```text
TAESD tiny decoder + LSGAN, 1k, seed 0
best_feasible_compliance = 167.2432
mean_top9                = 180.2190
mean_hamming             = 0.2160
runtime                  = 20:04
artifact                 = results/fem_cantilever_tiny_decoder_lsgan_seed0/top_designs_curiosity_0_seed_0.npz
```

## Launch Scripts

Use these from the repo root:

```bash
bash examples/topology_optimization/run_topopt_best_local.sh
bash examples/topology_optimization/run_topopt_ranker_sweep.sh
bash examples/topology_optimization/run_spark_best_topopt.sh
```

The Spark script is parameterized through environment variables:

```bash
N_ITER=20000 BATCH_SIZE=1024 BUFFER_MULTIPLIER=16 SEED=0 \
  bash examples/topology_optimization/run_spark_best_topopt.sh
```

Default Spark config is the current best learning setup:

```text
topk_volume + LSGAN + Conv G + MLP D + Muon/Muon + scheduled topology curiosity
curiosity=0.1, warmup_cosine, warmup_frac=0.05
generator_channels=64, latent_dim=64
```

## Caveats

- Results are single-seed unless noted; seed sweep is required before claiming robustness.
- Current FEM is CPU SciPy sparse solve; GPU only helps the neural nets unless a GPU FEM backend is integrated.
- `topk_volume` is discrete. The discriminator sees raw scores in the best setup; the evaluator applies top-k projection.
- `tiny_decoder` is MLP-only for now and does not support `--train_on_decoded` or topology-space curiosity.
- The local Muon implementation is experimental and self-contained in the example file.
- Do not compare `archive_best_compliance` blindly across different value-level layouts; use `best_feasible_compliance`.
