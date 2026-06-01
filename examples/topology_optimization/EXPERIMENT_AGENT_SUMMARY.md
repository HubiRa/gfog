# Topology Optimization Experiment Handoff

This summarizes the topology-optimization work so the next agent can continue on a larger NVIDIA machine without re-discovering the setup.

Latest concise findings are also captured in:

```text
examples/topology_optimization/TOPOLOGY_FINDINGS_MAY_2026.md
```

## Current Best Direction

There are two different benchmarks in this repo now. Do not mix the absolute
compliance numbers.

Latest standard 40x20 learning setup, as of 2026-05-28:

- `encoding=sorted_material`
- `sorted_material_profile=binary`
- `density_filter_radius=0`
- `projection_beta=0`
- `optimizer_type=quantile_ranked_default`
- `ranker_target_curve=exp`
- `ranker_tau=4`
- `ranker_weight=1.0`
- `ranker_sample_pool_size=128`
- `ranker_steps=1`
- `G=ConvDecoderGenerator`
- `D=MLP`
- `generator_output_norm=centered_l2`
- `G optimizer=Muon`
- `D optimizer=Muon`
- `g_lr=0.03`
- `d_lr=0.1`
- `curiosity_space=raw`
- `curiosity_reference=buffer`
- `curiosity=0.0003`
- `batch_size=64`
- `buffer_multiplier=8`

The centered-L2 output normalization is important for distance-based curiosity:
without output normalization, raw uniformity can be gamed by scale and previous
raw-curiosity results are not apples-to-apples. For sorted-material ordering,
centered-L2 preserves the ordering semantics while giving `D` and curiosity a
fixed-scale genome space.

Latest 3k multi-seed comparison:

```text
old baseline, no output norm, no curiosity:
seed0 73.6800
seed1 78.8317
seed2 74.9235
median 74.9235, mean 75.8117

centered_l2 + raw uniformity 0.0003:
seed0 72.5222
seed1 78.1218
seed2 74.2828
median 74.2828, mean 74.9756
```

Best individual 40x20 result so far is now:

```text
best_feasible_compliance = 72.5222
artifact = results/topopt_centered_l2_raw_uniformity_3k/quantile_tau4_iter3000_bs64_bufx8_bufdiv0_gnormcentered_l2_glr0.03_dlr0.1_curio0.0003_raw_batch_buffer_seed0/top_designs_curiosity_0.0003_seed_0.npz
```

Older learning setup that worked before output normalization:

- `encoding=topk_volume`
- `optimizer_type=quantile_ranked_default`
- `ranker_target_curve=exp`
- `ranker_tau=4`
- `ranker_weight=1.0`
- `ranker_sample_pool_size=128`
- `ranker_steps=1`
- `G=ConvDecoderGenerator`
- `D=MLP`
- `G optimizer=Muon`
- `D optimizer=Muon`
- `g_lr=0.03`
- `d_lr=0.03`
- `curiosity=0`

Do not compensate with multiple D/ranker steps by default. The best ranker
runs use one ranker update per iteration (`--ranker_steps 1`) and tune the
relative learning rates instead, following the TTUR-style idea. For the
ranked/quantile optimizers, `--discriminator_steps` is not the knob to tune;
the relevant discriminator/ranker update count is `--ranker_steps`, and the
current recommendation is to keep it at `1`.

The current best long-run discriminator is still the MLP. A listwise
set-transformer discriminator is implemented and promising at 1k, but it
stalled in the first 3k run.

Current best 40x20 result is the clean binary sorting variant:

```text
encoding=sorted_material
sorted_material_profile=binary
density_filter_radius=0
projection_beta=0
```

```text
Quantile-ranked vanilla GAN, exp tau=4, buffer=512, sample pool=128, 10k iters, seed 0
best_feasible_compliance = 72.9039
mean_top9                = 72.9340
relative best            = 1.8578
runtime                  = 1:19:36 on local CPU
```

Artifact:

```text
results/fem_cantilever_sorted_material_binary_quantile_tau4_iter10000_convG_mlpD_seed0/top_designs_curiosity_0_seed_0.npz
```

The 3k version already reached `74.2849`, so the 10k gain was modest but real.

Old 40x20 toy benchmark setting:

- `grid=40x20`
- `single center-right point load`
- `E=1`
- `volume_max=0.48`
- `density_filter_radius=1`
- `projection_beta=1`

Previous best completed 40x20 top-k/projection result:

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

Current TOM/GiNN-scale benchmark setting:

- `--preset tom_cantilever_2d`
- `grid=150x100`
- `domain=1.5x1.0`
- `E=196`
- two distributed right-edge traction patches
- `volume_max=0.48`
- `density_filter_radius=0`
- `solid_compliance=0.0038145905`

Best completed TOM/GiNN-scale result:

```text
Quantile-ranked vanilla GAN, exp tau=4, buffer=512, sample pool=128, 3k iters, seed 0, fem_workers=8
best_feasible_compliance = 0.0447193
mean_top9                = 0.0617
relative best            = 11.7232
runtime                  = 3:29:53 on local CPU
```

Artifact:

```text
results/fem_cantilever_tom_preset_quantile_tau4_iter3000_workers8_seed0/top_designs_curiosity_0_seed_0.npz
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

Recommended next Spark/DGX baseline for TOM/GiNN-scale comparison:

```text
Use --preset tom_cantilever_2d.
Start with batch_size=64, buffer_multiplier=8, ranker_sample_pool_size=128.
Use fem_workers=8 initially, then test 16/24/32 depending on CPU cores and memory.
Run n_iter=10000 first; 20000 only if the 10k curve is still improving.
```

Do not start TOM/GiNN-scale runs with `batch_size=1024`, `buffer_multiplier=16`.
That recommendation was for the old 40x20 toy benchmark. On the 150x100 TOM
preset it creates very expensive high-resolution sparse FEM batches.

Recommended command:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --preset tom_cantilever_2d \
  --n_iter 10000 --batch_size 64 --buffer_multiplier 8 \
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
  --fem_workers 8 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_tom_preset_quantile_tau4_iter10000_workers8_seed0
```

Old 40x20 Spark baseline, only if intentionally continuing the toy benchmark:

```text
Conv G + MLP D, quantile-ranked vanilla GAN, exp tau=4, top-tail ranker pool, Muon/Muon
n_iter: 10000 or 20000
batch_size: 1024
buffer_multiplier: 16
ranker_sample_pool_size: 2048 initially
effective elite buffer size: 16384
```

This old recommendation keeps the best learning setup and spends the larger
machine budget on a much larger elite buffer. It is only reasonable because the
40x20 topology tensors and FEM solves are small.

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
- `SetTransformerConvGenerator`, enabled with `--generator_type set_conv`.
  This lets latent samples communicate through a permutation-equivariant Transformer over the batch, then decodes each transformed token with the same convolutional decoder head.
  Optional stochastic elite genome context is available with `--set_generator_elite_context_size` and `--set_generator_elite_context_pool_size`; generated genome vectors are archived with their objective values, and future latent tokens cross-attend to sampled elite genomes before decoding.
- `SetTransformerDirectGenerator`, enabled with `--generator_type set_direct`.
  This removes the Conv decoder entirely: set-G emits the score/genome vector directly, the genome archive stores those emitted score vectors, and the existing black-box binary path (`sorted_material binary` / `topk_volume`) maps scores to topology.
- `ConvDiscriminator`, enabled with `--discriminator_type conv`.
- `SetTransformerDiscriminator`, enabled with `--discriminator_type set_transformer`.
  This treats the candidate batch/list as a set of tokens with no positional embeddings, applies self-attention across candidates, and emits one context-aware score per candidate.
  Useful for testing listwise D/ranker communication while keeping the Conv generator unchanged.
- Optional generic image decoder prior via `--encoding tiny_decoder`.
  This uses `diffusers.AutoencoderTiny` with `--tiny_decoder_model madebyollin/taesd` by default and projects the decoded grayscale image to the target top-k volume.
- Fixed-histogram sorting experiment via `--encoding sorted_material`.
  G emits only ordering scores; the evaluator assigns a fixed material histogram by sorted score. Profiles: `--sorted_material_profile binary|linear|sigmoid`.
- Ranked proposal-pool exploration for ranker optimizers:
  `--proposal_pool_size`, `--proposal_top_k`, `--proposal_diversity_min_hamming`, `--proposal_diversity_topk_frac`.
  This generates a larger G pool, scores it with D, keeps the top predicted candidates, then optionally applies greedy Hamming diversity before FEM evaluation.
- GA-style elite recombination for ranker optimizers:
  `--ga_offspring_fraction`, `--ga_pool_size`, `--ga_parent_pool_size`, `--ga_mutation_rate`, `--ga_mutation_scale`.
  This samples parents from the top buffer entries, creates uniform-crossover children in raw design-score space, applies sparse Gaussian mutation, scores the child pool with D, and replaces part of the evaluated batch with the selected children.
- G-only raw-output uniformity pre-warmup:
  `--g_uniformity_warmup_steps`, `--g_uniformity_warmup_batch_size`, `--g_uniformity_warmup_weight`, `--g_uniformity_warmup_t`.
  This updates only G with Wang-Isola uniformity before the initial buffer fill; no D/ranker/FEM updates occur during the warmup.
- TOM/GiNN-style cantilever preset via `--preset tom_cantilever_2d`.
  This sets a 150x100 grid on a 1.5x1 domain, `E=196`, `volume_max=0.48`, no density filter, and two right-edge traction patches matching the local TOM config.
- CPU-parallel SciPy FEM batch evaluation via `--fem_workers N`.
  This parallelizes independent compliance solves across worker threads while preserving batch order. It is supported for `--backend scipy`; keep `fem_workers=1` for `torchfem`.
- Optional connectivity objective inside `f` via `--connectivity_max` and `--connectivity_ladder`.
  This computes the fraction of solid cells not 4-connected to the left support and adds a connectivity violation before the usual volume/roughness/compliance objectives.
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
Sorted material binary, no filter/projection, tau=4, 10000 iters:    72.9039
Sorted material binary, no filter/projection, tau=4, 3000 iters:     74.2849
Sorted material binary, no filter/projection, tau=4, 3000, seed 1:   73.7453
Sorted material binary, no filter/projection, tau=4, 1000 iters:    102.4140
Sorted binary plain BCE GAN, no ranking, 1000 iters:                134.7742
Sorted binary plain LSGAN, no ranking, 1000 iters:                  178.9928
Sorted binary plain WGAN, no ranking, 1000 iters:                   146.9724
Sorted binary + set-transformer D, tau=4, 3000 iters:                83.0486
Sorted binary + set-transformer D, tau=4, 1000 iters:                83.5705
Sorted binary + set-transformer D + raw batch curiosity 0.001, 3k:   82.2937
Sorted binary + set-transformer D + raw batch curiosity 0.001, 1k:   82.2937
Sorted binary + setD + curiosity 0.001, list128/pool256, 3k:         80.1758
Sorted binary + setD + curiosity 0.001, list128/pool256, 1k:         81.8307
Sorted binary + set-transformer D + raw batch curiosity 0.01, 1k:    84.0322
Sorted binary + set-transformer D + proposal div0.05, 1000 iters:    82.3953
Sorted binary + set-conv G + set-transformer D, 1k, g_lr=0.03:      116.0943
Sorted binary + set-conv G + set-transformer D, 1k, g_lr=0.01:      113.9918
Sorted binary + set-conv G + set-D, Adam 3e-4, 3000 iters:         292.1855
Sorted binary + set-conv G + set-D, Adam 3e-4, 1000 iters:         321.9454
Sorted binary + set-conv G + set-D, Adam 1e-3, 1000 iters:         203.9757
Sorted binary + set-conv G + set-D + raw curiosity 0.003, 3k:       84.7169
Sorted binary + set-conv G + set-D + raw curiosity 0.005, 1k:      115.1713
Sorted binary + set-conv G + set-D + raw curiosity 0.003, 1k:       87.2996
Sorted binary + set-conv G + set-D + raw curiosity 0.001, 1k:       94.4134
Sorted binary + set-conv G + MLP-D, 1k:                            253.3375
Sorted binary + set-conv G + MLP-D + raw curiosity 0.003, 1k:       118.0784
Sorted binary + set-conv G + MLP-D + raw curiosity 0.005, 1k:       124.7300
Sorted binary + set-conv G + MLP-D + raw curiosity 0.01, 1k:        237.6787
Soft-volume sigmoid + set-conv G + MLP-D + raw curiosity 0.003, 1k: 275.8204
Sorted binary + set-conv G + set-D + genome ctx16/pool128 + cur0.003, 1k: 100.5963
Sorted binary + set-direct G + set-D + genome ctx16/pool128 + cur0.003, 1k: 4860.5063 infeasible
Sorted binary + set-direct G + MLP-D, 1k:                         3273.0559 infeasible
Sorted binary + set-direct G + MLP-D + connectivity_max=0, 1k:     3380.3057 infeasible
Coarse top-k 10x5 + set-direct G + MLP-D, 1k:                      139.2985
Sorted binary + set-conv G + set-D + elite ctx16/pool128 + cur0.003, 1k: 90.4264
Sorted binary + set-conv G + set-D + elite ctx4/pool32 + cur0.003, 1k:  137.0000
Sorted binary + proposal pool 512/top128/div0.05, 3000 iters:        78.9794
Sorted binary + proposal pool 512/top128/div0.05, 1000 iters:        81.0528
Sorted binary + proposal pool 512/top128/no diversity, 1000 iters:   86.2233
Sorted binary + GA 50%, pool256, mut2%, 1000 iters:                 146.6268
Sorted binary + GA 25%, pool512, mut0.5%, 1000 iters:               126.4075
Sorted binary + G-only uniformity warmup100, 1000 iters:            188.7710
Sorted material sigmoid24, no filter/projection, tau=4, 1000 iters:  111.1213
Sorted material linear, no filter/projection, tau=4, 1000 iters:     189.0848
```

Interpretation: listwise ranking is interesting conceptually, but the current PL implementations are not competitive with LSGAN. The best PL ranker result was archive Plackett-Luce with elite-margin G at `111.0036`, still behind Conv G LSGAN. Among the naive fake-rejection + PL-buffer variants, vanilla BCE GAN was best at `167.2213`, LSGAN was worse at `193.9727`, and WGAN was poor at `406.6346`. Curiosity made ranked LSGAN much worse by increasing diversity without usable compliance pressure.

The first non-PL dense-rank result is now the best overall direction: quantile-ranked vanilla GAN with exponential targets `exp(-rank / tau)` is very sensitive to `tau`. `tau=8` reached `100.5033` at 1k, matching the no-curiosity Conv G LSGAN 3k result, and improved to `92.1868` at 3k. Linear rank targets were poor at `227.4550`, `tau=4` was too sharp at `183.4090`, and softer `tau=16/32` were good but worse than `tau=8`.

Topology curiosity does not beat no-curiosity for quantile ranking in the 3k seed-0 sweep. `curiosity=1` nearly tied (`92.2906`) and had slightly higher diversity, but no curiosity remains best (`92.1868`). Small values were mixed: `0.01` nearly tied but collapsed harder, `0.005`/`0.03` were worse. Large values `10` and `100` are bad. Keep curiosity off for quantile ranking unless testing a different anti-collapse mechanism or much narrower schedules.

Batch-only topology curiosity was also bad for the current best quantile setting. `--curiosity_reference batch` avoids repelling from the elite buffer, but even `curiosity=0.001` degraded to `168.5401` and `0.01` degraded to `177.8041`. The Wang-Isola topology-uniformity term is therefore not the right anti-collapse mechanism for this setup, even batch-only.

The older top-k/projection current-best run crossed below the previous 91-ish bound: 10k iterations reached `90.7142` with mean top-9 `90.7837`. It improved only modestly from 3k (`91.3747`) and collapsed more (`mean_hamming=0.0066`). This is no longer the best 40x20 setting after the sorted-material binary runs.

Larger elite buffers did not help when ranker lists were sampled from the whole buffer. With batch `64`, buffer multiplier `2` means a buffer of `128` and reached `92.1868`. Buffer `256` degraded to `104.1730`; buffer `512` degraded to `109.4939`. The reason was likely rank-signal dilution. Adding `--ranker_sample_pool_size` fixed most of this: buffer `512` with ranker samples restricted to the top `128` reached `94.2259` at `tau=8`, and retuning to sharper `tau=4` reached the new best `91.3747`. Exact top-k ranker lists were worse (`106.8030`), so stochastic sorted sampling from the top-tail pool seems important. Pool-size/list-size ablations support this: pool `64`/list `64` was the same failure as exact top-k, pool `256`/list `64` was decent but worse (`96.3954`), and list `128` was very poor (`225.4730`). Global-rank targets were conceptually cleaner but worse: global `tau=16` got `101.9539`, global `tau=32` got `95.4780`. Keep local list ranks, list `64`, pool `128` for now.

Fixed-histogram sorting experiment:

```text
--encoding sorted_material
--density_filter_radius 0
--projection_beta 0
```

This tests the idea “fix the material amount/distribution and let G only sort where it goes.” The `binary` profile is equivalent to `topk_volume` without filter/projection and is now the best 40x20 result: `74.2849` at 3k seed 0, `73.7453` at 3k seed 1, and `72.9039` at 10k seed 0. Smooth fixed material histograms were worse in first tests: `sigmoid` with steepness 24 reached `111.1213`; `linear` reached `189.0848`. This suggests binary/near-binary material is still important for the current FEM objective. The second 3k seed being close to seed 0 is the best reproducibility signal so far; keep this as the default baseline.

Plain non-ranking GAN losses were tested on the same sorted-binary Conv-G/MLP-D setup with Muon/Muon and one discriminator step. They were much worse than the quantile-ranker: BCE GAN reached `134.7742`, WGAN reached `146.9724`, and LSGAN reached `178.9928` at 1k. They produced higher diversity but did not concentrate probability mass on useful low-compliance designs. Ranking is doing essential work here; plain real/fake GAN objectives are not enough for this black-box optimization loop.

Set-transformer discriminator:

```text
--discriminator_type set_transformer
--set_discriminator_dim 128
--set_discriminator_depth 2
--set_discriminator_heads 4
```

This tests a listwise/context-aware D: each candidate is a token, self-attention lets candidates compare, and D emits one score per candidate. No positional embeddings are used, so the module is permutation-equivariant over the candidate list. With sorted-binary + quantile tau=4 it reached `83.5705` at 1k, much better than the MLP-D no-proposal 1k baseline (`102.4140`) and close to proposal-pool MLP-D (`81.0528`). Increasing SetD context from `ranker_list_size=64, ranker_sample_pool_size=128` to `list=128, pool=256` helped: `81.8307` at 1k and `80.1758` at 3k. This is now the best set-containing run. It is still worse than the MLP-D 3k sorted-binary baseline (`74.2849`/`73.7453`), so SetD benefits from larger list context but is not the long-run default yet. Combining set-D with proposal-pool diversity reached `82.3953` at 1k, only a small gain over the smaller-context set-D run and worse than proposal-pool MLP-D.

Set-transformer D with G uniformity:

```text
--curiosity_space raw
--curiosity_reference batch
--curiosity 0.001
```

This applies Wang-Isola uniformity directly to generated raw score fields during the G update. `curiosity=0.01` increased diversity but slightly hurt compliance (`84.0322`, mean Hamming `0.0440`). `curiosity=0.001` was better: `82.2937` at 1k with mean Hamming `0.0533`, and a 3k run with smaller SetD context stayed at `82.2937` with mean Hamming `0.0622`. With larger SetD context (`list=128`, `pool=256`), the same curiosity improved further to `81.8307` at 1k and `80.1758` at 3k. So small raw batch uniformity plus larger list context is the best set-D recipe so far, but it remains behind the MLP-D sorted-binary baseline.

Set-transformer G:

```text
--generator_type set_conv
--set_generator_dim 128
--set_generator_depth 2
--set_generator_heads 4
```

This adds batch communication to G while preserving the Conv decoder head. Directly swapping it into the set-D sorted-binary tau=4 setup was worse: `116.0943` at 1k with Muon/Muon and `g_lr=0.03`, and `113.9918` with `g_lr=0.01`. The `g_lr=0.03` run produced high diversity (`mean_hamming=0.1138`) but poor compliance; reducing `g_lr` improved compliance only slightly and reduced diversity (`mean_hamming=0.0292`).

Adam-style Transformer learning rates were worse for this setup. Adam/Adam with `g_lr=d_lr=3e-4` reached only `321.9454` at 1k and `292.1855` at 3k, with the 3k run fully collapsed (`mean_hamming=0`). Adam/Adam at `1e-3` improved to `203.9757` at 1k but also collapsed (`mean_hamming=0`). Current interpretation: batch-aware G is easy to make diverse under Muon but not useful yet; with Adam it undertrains/collapses. It needs a better training signal/schedule, not just a direct optimizer swap.

Adding raw batch uniformity to set-G helped a lot but was still not competitive. With Muon/Muon and `g_lr=d_lr=0.03`, `curiosity=0.001` reached `94.4134` at 1k, `curiosity=0.003` reached `87.2996` at 1k and `84.7169` at 3k, while `curiosity=0.005` fell back to `115.1713`. The best set-G result so far is therefore `84.7169`, still behind Conv-G + set-D at 1k (`83.5705`) and far behind the independent Conv-G + MLP-D sorted-binary 3k baseline (`74.2849`).

Set-G with MLP-D was also tested. Without uniformity it was poor (`253.3375` at 1k). Adding raw batch uniformity `0.003` improved it to `118.0784`, but stronger values hurt: `0.005` reached `124.7300` and `0.01` reached `237.6787` with very high diversity (`mean_hamming=0.4097`). This is still worse than set-G + set-D + uniformity (`87.2996`) and worse than Conv-G + MLP-D sorted-binary (`102.4140` at 1k). So for set-G, setD remains the better ranker in early runs, and uniformity has a narrow useful range.

Replacing sorting with sigmoid/soft-volume decoding inside `f` was worse for set-G + MLP-D. Using `--encoding soft_volume` with `curiosity=0.003` reached only `275.8204` at 1k. It was feasible by volume/roughness, but much worse than the sorted-binary counterpart (`118.0784`). The run also produced sigmoid overflow/singular-matrix warnings, indicating extreme logits / near-degenerate material layouts.

Elite-context set-G was added to better match the “evolution proxy” idea. The first implementation exposed raw elite design/score vectors to G:

```text
--set_generator_elite_context_size 16
--set_generator_elite_context_pool_size 128
```

This lets latent tokens attend to randomly sampled top-buffer elite design tokens before the Conv decoder. With `curiosity=0.003`, ctx16/pool128 reached `90.4264` at 1k, worse than the no-context set-G curiosity run (`87.2996`). A smaller/more local ctx4/pool32 run was much worse (`137.0000`). Current interpretation: giving G direct elite tokens adds the missing stochastic recombination input, but naive attention over raw elite score fields mostly encourages copying/noisy conditioning, not useful structural recombination. If revisiting, the context should probably be encoded as decoded binary/topology masks or local patches/components, not raw score vectors.

This was then corrected to genome-level context: set-G stores the generated post-transformer genome vector for each evaluated proposal in a `GenomeArchive`, ranked by true objective values. Future random latent queries cross-attend to sampled elite genomes, not raw topology/design tensors. This matches the intended “informed generation with optional recombination” model more closely:

```text
z query tokens + sampled elite genome memory -> cross-attention -> child genome -> Conv decoder
```

However, the first genome-context run was still worse: ctx16/pool128 + `curiosity=0.003` reached `100.5963` at 1k. That is worse than no-context set-G (`87.2996`) and raw-design context (`90.4264`). Current interpretation: the mechanism is now semantically right, but the stored genome is a drifting internal representation tied to changing decoder weights. A more stable explicit genome/decoder split may be needed if we continue this direction.

Retesting the same genetic/genome context with set-G + set-D and the larger SetD ranker context (`ranker_list_size=128`, `ranker_sample_pool_size=256`) was worse, not better: ctx16/pool128 + `curiosity=0.003` reached `147.5608` at 1k with high diversity (`mean_hamming=0.2258`) and poor mean top-k (`290.9736`). Current interpretation: elite genome cross-attention injects variation, but the variation is not aligned with useful structural recombination. Do not treat this as a promising path unless the context is made more stable/local, e.g. explicit patch/component context rather than drifting post-transformer genomes.

Set-direct G tested that stable split by omitting the Conv decoder entirely:

```text
--generator_type set_direct
--encoding sorted_material
--sorted_material_profile binary
```

Here the emitted score vector is both the genome and the black-box input; the binary/sorted-material projection creates the topology. The first run failed badly: ctx16/pool128 + `curiosity=0.003` reached `4860.5063` best archive compliance and was infeasible due to high roughness violation. The direct binary black-box path appears to produce very noisy/disconnected rankings without a spatial decoder or explicit smoothness/locality bias. This reinforces that the Conv decoder is currently doing useful spatial regularization, even if it makes genome archiving less clean.

A cleaner no-decoder baseline with `set_direct + MLP-D + sorted_material binary` also failed: `3273.0559` at 1k and infeasible due to roughness violation. Moving locality into the black box with `--encoding coarse_topk_volume --coarse_grid_width 10 --coarse_grid_height 5` was feasible and much better (`139.2985` at 1k), but still far behind the Conv-G sorted-binary baseline. Increasing the coarse genome to 20x10 made the same setup much worse (`1220.2235` at 1k, feasible but poor, `mean_hamming=0.0253`). This suggests the clean coarse-genome formulation works mechanically, but direct high-dimensional set-G search does not scale just by adding more cells.

```text
G -> low-res genome g
f -> upsample/sort/project g to binary topology and evaluate
B -> stores (g, s)
```

It preserves the rule that decoder/projection lives only inside `f`, but the current coarse genome needs more work.

Current clean-genome interpretation: 10x5 coarse top-k is the best no-decoder result so far because it injects locality and a strong dimensionality bottleneck inside `f`. The 20x10 run removes too much of that bottleneck and leaves the ranker/G pair with a harder, mostly discrete search problem. Do not spend long runs on larger coarse grids until the generator has a better locality prior, mutation/recombination operator, or staged coarse-to-fine mechanism.

Connectivity objective:

```text
--connectivity_max 0
```

This adds a black-box objective for disconnected material: fraction of solid cells not 4-connected to the left support. It is prepended before the final volume/roughness/compliance levels, so the buffer sorts disconnected designs behind connected ones. On flat `set_direct + sorted_material binary`, it worked mechanically but did not solve the problem: best vector started with connectivity violation `0.359375` and compliance `3380.3057`, still infeasible. So the objective can punish disconnected islands, but the flat genome/search setup still struggles to discover connected structures. Coarse/local genomes remain the better clean direction.

Ranked proposal-pool exploration:

```text
--proposal_pool_size 512
--proposal_top_k 128
--proposal_diversity_min_hamming 0.05
```

This improved the 1k sorted-binary run substantially (`81.0528` vs `102.4140`). Pure D-pool preselection without Hamming diversity also helped but less (`86.2233`). The 3k diversity run underperformed the no-proposal 3k baseline (`78.9794` vs `74.2849`), so this is useful for early exploration but not yet a better long-run default. Next tests: anneal/disable diversity after warmup, or use a smaller threshold such as `0.01`/`0.02`.

Proposal-pool evolution was then added: generate a large Conv-G proposal pool, optionally replace part of that pool by crossover/mutation children, rank everything with D, apply optional Hamming diversity, and FEM-evaluate only the selected batch. This keeps Conv-G as the spatial prior and treats recombination as a non-differentiable proposal operator.

First results were mixed/negative:

```text
proposal_pool=512 top_k=128 diversity=0.05
evolution_fraction=0.5 parent_source=pool_buffer crossover=row mutation=0.01/0.1
1k best_feasible_compliance = 155.6045

proposal_pool=512 top_k=128 diversity=0.05
evolution_fraction=0.25 parent_source=pool crossover=rect mutation=0.005/0.05
1k best_feasible_compliance = 86.2149
```

The light pool-only variant is feasible and better than the plain 1k sorted-binary baseline (`102.4140`), but it is essentially tied with no-diversity proposal preselection (`86.2233`) and worse than proposal preselection with diversity (`81.0528`). Current interpretation: D-ranked oversampling is useful; raw score-map recombination is not yet adding value. If continuing, use much weaker local operators or anneal evolution off after early exploration rather than making it a default.

Component-aware binary head:

```text
--binhead_connect_support
```

This adds deterministic black-box postprocessing after binary top-k/sorted decoding and before FEM: keep material connected to the left support, then refill removed cells by growing into high-score neighboring cells until the target volume is restored. It does not affect G/D gradients and only changes `f`.

First sorted-binary Conv-G 1k result was worse: `108.4622` vs the plain 1k baseline `102.4140`. Diversity increased (`mean_hamming=0.0642`), but compliance degraded. The same repair with set-G + set-D + raw batch curiosity `0.003` reached `90.8329` at 1k, worse than the no-binhead set-set baseline (`87.2996`). Current interpretation: this naive repair preserves volume/connectivity but disrupts useful load paths. Do not run long versions unless the repair is made softer/local or exposed as an objective rather than forced postprocessing.

D-gradient proposal refinement:

```text
--proposal_gradient_steps 1
--proposal_gradient_step_size 0.05 or 0.01
--proposal_gradient_keep_original
```

This uses the trained discriminator as a local learned fitness landscape: after G emits proposal score maps, take a gradient-ascent step on `D(proposal)` with respect to the proposal tensor, then D-rank original plus refined proposals and FEM-evaluate only the selected batch. It keeps G training unchanged and only changes proposal selection/evaluation.

First SetD tests were negative. On the best SetD recipe (`Conv G + SetD + sorted binary + curiosity=0.001 + list128/pool256`), baseline 1k was `81.8307`. One normalized continuous D-gradient step with step size `0.05` reached `88.8363`; step size `0.01` reached `89.7709`. A projection-aware swap mode was then added:

```text
--proposal_gradient_mode swap
```

This decodes the current top-k material set, uses `dD/dg` to demote low-gradient solid cells and promote high-gradient void cells, and preserves the material count. It was better than continuous ascent but still did not beat baseline: swapping `2%` of material cells reached `82.7492`; swapping `0.5%` reached `88.4095`; swapping `5%` reached `83.5075`. Current interpretation: the gradient signal contains some useful local ranking information, but SetD is not calibrated enough for gradient-guided mutation to improve FEM results directly. The best swap size tested was `2%`, but even that is worse than no swap. If revisiting, use D-gradient swaps only inside a larger proposal pool with FEM/D double selection, or train D with an explicit local perturbation consistency objective.

GA-style elite recombination:

```text
--ga_offspring_fraction 0.5
--ga_pool_size 256
--ga_parent_pool_size 128
--ga_mutation_rate 0.02
--ga_mutation_scale 0.25
```

This reached only `146.6268` at 1k. A lighter variant with `--ga_offspring_fraction 0.25`, `--ga_pool_size 512`, `--ga_mutation_rate 0.005`, `--ga_mutation_scale 0.1` improved to `126.4075` but was still worse than the no-GA sorted-binary 1k baseline (`102.4140`). Current interpretation: naive raw-score uniform crossover produces children that are feasible but not aligned with the learned generator/ranker dynamics. If revisiting GA, try topology-aware operators on decoded masks, small local material swaps, or insert GA children only into the buffer as auxiliary evaluations rather than replacing a large fraction of the G proposal batch.

G-only uniformity pre-warmup:

```text
--g_uniformity_warmup_steps 100
--g_uniformity_warmup_batch_size 256
--g_uniformity_warmup_weight 10
--curiosity 0
```

This made initial raw G outputs much more diverse, but the first 1k sorted-binary run was poor: `188.7710` best feasible compliance, mean top-9 `275.9148`, mean Hamming `0.2928`. Interpretation: raw-output uniformity alone spreads the generator but does not align the spread with useful topology structure, and the ranker did not recover within 1k iterations. If revisiting this idea, use a much weaker/shorter warmup or combine it with proposal-pool exploration rather than treating raw uniformity as a standalone initialization.

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

## Post-TSP Topology Lessons Encoded

The TSP experiments made the representation lesson explicit: GFog can refine
a useful prior, but it burns evaluations when the search space has weak local
meaning. For topology optimization, keep the search spatial and physics-aware:

- Default back to spatial `ConvDecoderGenerator`; do not prioritize flat genomes.
- Keep `ranker_steps=1`; tune `g_lr`/`d_lr` instead of doing multiple D/ranker steps.
- Compare the old quantile fake-rejection objective against the cleaner mixed
  evaluated ranker:
  `--optimizer_type hybrid_contextual_utility`.
- For mixed ranking, `D` trains on a sorted list containing current buffer
  entries plus the latest truly evaluated `G` outputs. This avoids disconnecting
  `D` from the distribution that `G` actually visits.
- Track behavior, not just endpoints. History arrays now include
  `best_feasible_last`, `feasible_rate`, mean volume/roughness violations, and
  a connectivity-violation column when present.
- Test constraint logic inside `f`, not in the trainable decoder. The new suite
  includes a connectivity-objective variant and a deterministic connect-repair
  variant for binary heads.

Reusable local/Spark suite:

```bash
bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
```

Default suite:

```text
quantile_best              current sorted-binary quantile baseline
mixed_ranker               mixed evaluated contextual ranker, no curiosity
mixed_ranker_curiosity     mixed ranker plus small batch-only curiosity
mixed_ranker_connectivity  mixed ranker with connectivity violation in f
```

Default local settings:

```text
grid=40x20
encoding=sorted_material
sorted_material_profile=binary
density_filter_radius=0
projection_beta=0
G=conv
D=mlp
Muon/Muon
g_lr=d_lr=0.03
batch=64
buffer_multiplier=8
ranker_list_size=64
ranker_sample_pool_size=128
n_iter=1000
```

Useful overrides:

```bash
N_ITER=3000 FEM_WORKERS=8 \
EXPERIMENTS="quantile_best mixed_ranker mixed_ranker_curiosity mixed_ranker_connectivity" \
bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
```

For the older `topk_volume` setup where topology-space curiosity is supported:

```bash
ENCODING=topk_volume DENSITY_FILTER_RADIUS=1 PROJECTION_BETA=1 \
CURIOSITY_SPACE=topology \
EXPERIMENTS="quantile_best mixed_ranker mixed_ranker_curiosity" \
bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
```

Same-size force-distribution suite:

```bash
bash examples/topology_optimization/run_topopt_loadcase_suite.sh
```

This keeps the current best 40x20 sorted-binary quantile setup fixed and varies
only the right-edge force vector in `f`:

```text
center_point
right_top_point
right_bottom_point
right_two_points
right_edge_uniform
right_edge_shear
```

The goal is to check whether the optimizer/representation works only for the
single center-point load or transfers across load distributions on the same
mesh/support/volume budget.

1. TOM/GiNN-scale continuation:

```text
baseline: --preset tom_cantilever_2d
n_iter: 10000 first, then 20000 only if still improving
batch_size: 64 initially
buffer_multiplier: 8
ranker_sample_pool_size: 128
fem_workers: 8 initially; test 16, 24, 32
```

2. TOM/GiNN-scale worker and batch throughput:

```text
Keep the learning setup fixed and measure wall-clock per 100 iterations.
Try fem_workers: 8, 16, 24, 32
Then try batch_size: 128, 256 only if memory and sparse-solve throughput are stable.
Do not jump to batch_size=1024 on the TOM preset until measured.
```

3. TOM/GiNN-scale ranker pool after batch changes:

```text
batch_size=64: ranker_sample_pool_size=128
batch_size=128: ranker_sample_pool_size=256
batch_size=256: ranker_sample_pool_size=512
Keep ranker_list_size=64 initially.
```

4. Seed sweep after a stable TOM/GiNN setting:

```text
seeds: 0, 1, 2, 3, 4
n_iter: 3000 and 10000
Only do this after checking the 10k learning curve.
```

5. Old 40x20 toy benchmark, only if intentionally continuing that benchmark:

```text
batch_size: 1024, 2048, 4096
buffer_multiplier: 16
ranker_sample_pool_size: 2048 initially
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

Warning: `run_spark_best_topopt.sh` was written for the old 40x20 small-grid
benchmark unless it has been updated on the target machine. For TOM/GiNN-scale
runs, prefer the explicit `--preset tom_cantilever_2d` command above.

The Spark script is parameterized through environment variables:

```bash
N_ITER=20000 BATCH_SIZE=1024 BUFFER_MULTIPLIER=16 SEED=0 \
  bash examples/topology_optimization/run_spark_best_topopt.sh
```

Default Spark script config, if unchanged, is historical:

```text
topk_volume + LSGAN + Conv G + MLP D + Muon/Muon + scheduled topology curiosity
curiosity=0.1, warmup_cosine, warmup_frac=0.05
generator_channels=64, latent_dim=64
```

This is not the latest TOM/GiNN-scale recommendation. The current recommended
Spark/DGX run is explicit:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --preset tom_cantilever_2d \
  --n_iter 10000 --batch_size 64 --buffer_multiplier 8 \
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
  --fem_workers 8 \
  --seed 0 \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_tom_preset_quantile_tau4_iter10000_workers8_seed0
```

## Caveats

- Results are single-seed unless noted; seed sweep is required before claiming robustness.
- Current FEM is CPU SciPy sparse solve; GPU only helps the neural nets unless a GPU FEM backend is integrated.
- TOM/GiNN-scale runs should use `--preset tom_cantilever_2d`; do not compare their absolute compliance to the old 40x20 toy setting.
- `--fem_workers` parallelizes SciPy FEM batch evaluation. Retune worker count on the target machine.
- `topk_volume` is discrete. The discriminator sees raw scores in the best setup; the evaluator applies top-k projection.
- `tiny_decoder` is MLP-only for now and does not support `--train_on_decoded` or topology-space curiosity.
- The local Muon implementation is experimental and self-contained in the example file.
- Do not compare `archive_best_compliance` blindly across different value-level layouts; use `best_feasible_compliance`.

## Latest `volume_max=0.35` Reduced-Material Findings

The reduced-material 40x20 center-load benchmark is now the most useful small
stress test. Continuous OC is very strong here:

```text
continuous SIMP/OC, no filter: 107.4058
binarized OC reference:        about 110.56
best GFog run:                 109.3426
```

Best GFog run:

```text
optimizer_type=quantile_ranked_default
encoding=sorted_material
sorted_material_profile=binary
G=conv, G_norm=centered_l2
D=mlp
Muon/Muon
g_lr=0.07
d_lr=0.12
batch_size=128
buffer_multiplier=4
ranker_tau=4
ranker_list_size=64
ranker_sample_pool_size=128
curiosity=0.0003 raw-vs-buffer
n_iter=1000
seed=2
best_feasible_compliance=109.3426
```

Artifact:

```text
results/topopt_vol035_lr_seed_neighborhood_1k/
  quantile_ranked_default_tau4_iter1000_vol0.35_bs128_bufx4_...
  _glr0.07_dlr0.12_curio0.0003_..._seed2/
  top_designs_curiosity_0.0003_seed_2.npz
```

Robust LR sweep, three seeds at 1k:

```text
g=0.07 d=0.08: 124.00 / 113.11 / 117.84, mean 118.32
g=0.07 d=0.12: 127.76 / 121.42 / 109.34, mean 119.51
g=0.06 d=0.08: 134.93 / 126.03 / 121.84, mean 127.60
g=0.06 d=0.10: 114.65 / 524.93 / 135.20, mean 258.26
```

Conclusion: the old `g=0.06, d=0.10` value was a sharp seed-0 win, not a
robust baseline. For future `vol=0.35` work, start with `g=0.07, d=0.08` for
robustness and `g=0.07, d=0.12` for upside.

Negative robustness tests:

```text
D-only value auxiliary:
  utility_weight=0.001: 405.76
  utility_weight=0.003: 114.20
  utility_weight=0.01:  124.96
  utility_weight=0.1:   579.96

fake target weight, one rank list:
  fakew=0:   2422.17
  fakew=0.1: 2422.17
  fakew=0.3: 1619.97
  fakew=1:    114.65

fake batch repeats:
  frep=1: 114.65
  frep=2: 130.14
  frep=4: 117.26
  frep=8: 170.50
```

Lessons:

- Keep G trained only through the rank/fake discriminator signal; do not feed
  direct compliance or value targets into G.
- The shared D value head did not stabilize the system. It added gradient
  conflict and made performance more sensitive.
- Full `D(fake)=0` is essential. Weakening it lets G exploit uncalibrated
  off-buffer score holes.
- More fake batches or more rank-list averaging per D update did not improve
  robustness.
- The useful lever remains TTUR-style LR tuning, not extra D losses or extra D
  steps.

OC-like method limits:

```text
OC is excellent for classic SIMP compliance + volume constraints.
GFog is more compelling when f is cheap to evaluate in large batches but hard,
discrete, black-box, or unreliable to differentiate.
```

Promising non-OC-friendly topopt classes include black-box simulator objectives,
hard binary/manufacturing constraints, connectivity and component rules,
stress/buckling/contact/fracture/fatigue/crash constraints, robust stochastic
load/material cases with non-smooth aggregation, and learned/human-defined
objectives. Many are still gradient-solvable with adjoints/MMA/SQP if
sensitivities exist; the target niche for GFog is where those sensitivities are
unavailable or not worth engineering.

Implemented next local benchmark:

```bash
bash examples/topology_optimization/run_topopt_robust_multiload.sh
```

This uses the existing SciPy FEM cantilever but evaluates each topology under
multiple load cases via `--robust_load_cases` and ranks by
`--robust_load_aggregate max|mean|cvar`. Default script setting is the current
reduced-volume quantile-ranker baseline: sorted binary material, Conv G, MLP D,
Muon/Muon, `g_lr=0.07`, `d_lr=0.08`, `volume_max=0.35`, `batch_size=128`,
`buffer_multiplier=4`, `curiosity=0.0003`, and robust max over
`center_point right_top_point right_bottom_point right_edge_uniform`.

First completed robust sanity run used the same setting but `batch_size=64` and
`n_iter=1000`. It reached best aggregate max compliance `372.1453`, mean top-9
`392.1672`, mean Hamming `0.0386`, runtime `2:54` local. The best design's
individual compliances were `348.9781` center, `372.1453` top, `363.2189`
bottom, and `351.4646` uniform, so the aggregate was genuinely balancing load
cases. Artifact:

```text
results/topopt_robust_multiload_1k/robust_max_center_point-right_top_point-right_bottom_point-right_edge_uniform_iter1000_vol0.35_bs64_bufx4_gnormcentered_l2_muon_muon_glr0.07_dlr0.08_curio0.0003_seed0/top_designs_curiosity_0.0003_seed_0.npz
```

External benchmarks to try later:

- PyTOPress: Python/NumPy/SciPy design-dependent pressure-load topology
  optimization. Attractive because the load changes with the topology.
- pyMOTO examples: stress constraints, overhang filters, robust formulations,
  eigenfrequency, transient thermal, self-weight, compliant mechanisms. Wrap as
  black-box `f` for GFog comparisons.

Latest staged material-removal objective:

```bash
bash examples/topology_optimization/run_topopt_removal_ladder.sh
```

G emits one priority score per element. `f` keeps the top-k material at each
configured volume fraction and returns lexicographic compliance-threshold
violations plus the normal `volume, roughness, compliance` summary tail. First
test used volumes `0.50 0.45 0.40 0.35 0.30` with thresholds
`80 95 115 145 190` on the center-point load. Robust-load LR
`g=0.01,d=0.1` failed the first stage. Single-load LR `g=0.07,d=0.08` at 1k
solved the first three stages:

```text
vol=0.50  C=78.2 / 80
vol=0.45  C=92.6 / 95
vol=0.40  C=113.6 / 115
vol=0.35  C=243.7 / 145
vol=0.30  C=5390.7 / 190
```

Artifact:

```text
results/topopt_removal_ladder_1k_glr007_dlr008/removal_center_point_vols0.50-0.45-0.40-0.35-0.30_iter1000_bs64_bufx4_gnormcentered_l2_muon_muon_glr0.07_dlr0.08_curio0.0003_seed0/removal_ladder_stages_seed0.png
```

Connectivity variants were added and tested:

- `--removal_ladder_connectivity_max X` inserts a disconnected-material
  violation before each stage compliance violation.
- `REMOVAL_CONNECT_REPAIR=true` passes `--binhead_connect_support` and applies
  support-connect/refill repair inside each removal stage before FEM.

Results so far are negative. Strict `--removal_ladder_connectivity_max 0` at
500 iters produced `[0.0, 68.5, 0.0389, 313.7, ...]`: first stage connected,
but compliance still bad and later stages disconnected. Repair was both slow
and poor: `100` iters with `bs=32` took `3:28` and failed the first compliance
stage badly. Do not use repair for larger sweeps until optimized; prefer a
shorter/adaptive ladder and only then reintroduce connectivity.

Adaptive 35% removal ladder improved substantially:

```text
volumes:      0.50 0.45 0.40 0.35
thresholds:  80   95   125  220
setting:     g_lr=0.07, d_lr=0.08, 1k iters, bs=64
result:      C35=151.5835, all stage violations zero
```

Stage compliances were `78.4`, `92.6`, `115.6`, `151.6`. This is the current
best removal-ladder result and supports adaptive thresholding: make stages
reachable first, then tighten.

Follow-up threshold tightening:

```text
thresholds 80 95 125 180 -> same C35=151.5835
thresholds 80 95 125 160 -> same C35=151.5835
```

So tightening only the final `0.35` cap did not improve the trajectory once the
stage was reachable. Next test should add an intermediate lower-volume stage,
for example volumes `0.50 0.45 0.40 0.35 0.325`, before returning to `0.30`.

Direct fixed-volume comparison at `volume_max=0.35`, same model/LR, 1k iters:

```text
seed0: C35=128.4664
seed1: C35=529.5534
seed2: C35=413.1860
```

So direct fixed-volume has better upside than removal ladder (`128.5` vs
`151.6`) but worse seed stability in this quick comparison. Removal ladder is
currently more useful as a nested-ordering/robustness experiment than as the
best way to optimize a single final volume.

Degenerate compliance-only objective with official `Levels.ladder` support:

```text
script: examples/topology_optimization/run_topopt_ttur_sweep.sh
extra env: LEVELS_LADDER="compliance:180,160,140"
           LEVELS_LADDER_FINAL_OPEN=compliance
same setting: Conv G / MLP D, spectral D, Muon/Muon,
              g_lr=0.07, d_lr=0.08, curiosity=0.0003,
              batch_size=64, buffer_multiplier=4, 1k iters

seed0: C35=126.0838, mean top9=126.7317
seed1: C35=529.5534, mean top9=532.2278
seed2: C35=413.1860, mean top9=422.7203
```

Interpretation: this is useful infrastructure but not a major new signal. A
single-objective compliance ladder is nearly monotone-equivalent to raw
compliance, so it does not solve bad-seed collapse. It can slightly change the
training dynamics and gave the best 1k lucky-seed fixed-volume result so far
(`126.08`), but robustness needs a nontrivial staged signal such as changing
volume/material constraints or an auxiliary objective.

Actual material+compliance ladder using the same usual Conv-G/MLP-D ranker
machinery requires variable material. Fixed `sorted_material`/`topk_volume`
make volume constant, so a volume ladder is a no-op there. The meaningful test
used direct hard-binary decoding:

```text
ENCODING=direct
HARD_BINARIZE=true
G=conv, D=mlp, spectral D, Muon/Muon
g_lr=0.07, d_lr=0.08
batch_size=64, buffer_multiplier=4
curiosity=0.0003
LEVELS_LADDER="volume:min:0.50,0.45,0.40,0.35 compliance:min:300,260,220,180"
LEVELS_LADDER_FINAL_OPEN=compliance
RUN_TAG=mat_comp_ladder

centered_l2 G norm, 500 iters:
  top volumes about 0.42-0.45
  no volume<=0.35 feasible sample
  best_any_compliance=222.8466

l2 G norm, 500 iters:
  top volumes about 0.3425-0.3500
  best_feasible_compliance=155.0014

l2 G norm, 1000 iters:
  top volumes about 0.3438-0.3500
  best_feasible_compliance=151.8213
  mean top9=153.4096
```

Interpretation: this is the intended staged objective class, not the
compliance-only test. It works mechanically and discovers reasonable truss-like
designs, but it is not competitive yet with fixed sorted-binary optimization.
Important lesson: material ladders need a global material-control direction in
the genome. `centered_l2` removes that direction; `l2` preserves bounded scale
while still allowing output sign/mean to control material usage.

Coarse top-k symmetry-reduction test:

```text
Hypothesis:
  sorted/top-k has huge permutation symmetry. Let G emit a lower-res score grid,
  upsample it to 40x20, then use the usual full-grid top-k projection.

Script support:
  examples/topology_optimization/run_topopt_ttur_sweep.sh now exposes
  ENCODING, COARSE_GRID_WIDTH, and COARSE_GRID_HEIGHT.

Common setting:
  encoding=coarse_topk_volume
  volume_max=0.35
  Conv G / MLP D, spectral D, Muon/Muon
  centered_l2 G output norm
  batch_size=64, buffer_multiplier=4
  curiosity=0.0003
  n_iter=1000

20x10, g_lr=0.07, d_lr=0.08:
  best_feasible_compliance=486.9804
  mean top9=487.8121
  mean_hamming=0.0068

10x5, g_lr=0.07, d_lr=0.08:
  best_feasible_compliance=549.5952
  mean top9=552.8832
  mean_hamming=0.0079

20x10, g_lr=0.03, d_lr=0.10:
  best_feasible_compliance=521.5104
  mean top9=524.8285
  mean_hamming=0.0150
```

Interpretation: plain coarse top-k is a negative drop-in replacement. It
reduces sorting symmetry, but the bilinear-upsampled score field plus hard top-k
threshold creates smooth single-blob bands. The displayed `20x10` result was a
horizontal band, not a truss/diagonal load path. If continuing the symmetry
idea, use a coarse-to-fine residual or patch hierarchy rather than a pure coarse
score field.

Non-sorting alternatives tested:

```text
common:
  volume_max=0.35
  quantile_ranked_default, tau=4
  batch_size=64, buffer_multiplier=4
  curiosity=0.0003
  n_iter=1000, seed=0

soft_volume, layernorm, projection_beta=0:
  best_feasible_compliance=386.6098
  mean top9=387.5450

soft_volume, layernorm, projection_beta=8:
  best_feasible_compliance=217.4709
  mean top9=218.3932

soft_volume, layernorm, hard_binarize=true:
  best_feasible_compliance=247.2338
  mean top9=252.6333

coarse level-set ladder:
  encoding=coarse, code_grid=20x10, hard_binarize=true, l2 G norm
  material/compliance ladder
  no volume<=0.35 feasible sample
  best_any_compliance=275.3018

bar primitives, 16 bars:
  encoding=bar_primitives
  G=mlp, D=mlp, no G output norm
  bar_count=16
  width=[0.015,0.06]
  edge_softness=0.008
  g_lr=0.01, d_lr=0.03
  material/compliance ladder
  best_feasible_compliance=158.2036
  mean top9=161.4743
  mean_hamming=0.1265

bar primitives, 24 thinner bars:
  best_feasible_compliance=183.3318
  mean top9=188.3109

bar primitives, 16 bars, hard_binarize=true:
  no fully feasible sample due to roughness violation
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

Implementation note: `bar_primitives` is now an `f`-side encoding. G emits
`bar_count * 5` raw parameters: two endpoints and one width logit per bar.
The black box maps endpoints through sigmoid to normalized coordinates,
maps width into `[bar_width_min, bar_width_max]`, rasterizes line segments to
the 40x20 density grid, then evaluates FEM. No sorting/top-k is used. The
16-bar result is visually truss-like and is the best non-sorting structured
representation so far, but it still trails direct hard-binary material ladder
(`151.8`) and fixed sorted binary. LR tuning did not beat `g_lr=0.01,d_lr=0.03`;
`g_lr=0.03,d_lr=0.01` was close but with lower diversity. Next tests should
constrain/seed anchors near support/load or add bar-connectivity priors, not
just add more bars.
