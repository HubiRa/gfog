#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-1000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-8}"
FEM_WORKERS="${FEM_WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-results}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

COMMON_ARGS=(
  --preset tom_cantilever_2d
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding sorted_material
  --sorted_material_profile binary
  --initial_buffer_mode random_blobs
  --initial_blob_count_max 8
  --initial_blob_radius_min 0.04
  --initial_blob_radius_max 0.30
  --initial_blob_min_hamming 0.2
  --density_filter_radius 0
  --projection_beta 0
  --generator_type conv
  --generator_output_norm centered_l2
  --generator_channels 64
  --discriminator_type mlp
  --discriminator_hidden_dims 128 128
  --optimizer_type quantile_ranked_default
  --ranker_weight 1.0
  --ranker_target_curve exp
  --ranker_tau 4
  --ranker_target_scope local
  --ranker_steps 1
  --ranker_sample_pool_size 128
  --ranker_sample_mode random_top_pool
  --discriminator_steps 1
  --curiosity 0.0003
  --curiosity_space raw
  --curiosity_reference buffer
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --fem_workers "${FEM_WORKERS}"
  --history_interval 50
  --seed "${SEED}"
)

LR_SPECS=(
  "g0p01_d0p03 0.01 0.03"
  "g0p03_d0p03 0.03 0.03"
  "g0p03_d0p10 0.03 0.10"
  "g0p06_d0p10 0.06 0.10"
)

for spec in "${LR_SPECS[@]}"; do
  read -r label g_lr d_lr <<< "${spec}"
  uv run python examples/topology_optimization/cantilever_fem.py \
    "${COMMON_ARGS[@]}" \
    --g_lr "${g_lr}" \
    --d_lr "${d_lr}" \
    --output_dir "${OUT_ROOT}/tom_cantilever_blobinit_nosmooth_lr_${label}_iter${ITER}_seed${SEED}"
done
