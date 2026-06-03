#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-500}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
FEM_WORKERS="${FEM_WORKERS:-8}"
GAN_OPTIMIZER_TYPE="${GAN_OPTIMIZER_TYPE:-default}"
CURIO="${CURIO:-0}"
OUT_ROOT="${OUT_ROOT:-results}"
INITIAL_BUFFER_MODE="${INITIAL_BUFFER_MODE:-generator}"
INITIAL_BLOB_COUNT_MIN="${INITIAL_BLOB_COUNT_MIN:-1}"
INITIAL_BLOB_COUNT_MAX="${INITIAL_BLOB_COUNT_MAX:-8}"
INITIAL_BLOB_RADIUS_MIN="${INITIAL_BLOB_RADIUS_MIN:-0.04}"
INITIAL_BLOB_RADIUS_MAX="${INITIAL_BLOB_RADIUS_MAX:-0.30}"
INITIAL_BLOB_MIN_HAMMING="${INITIAL_BLOB_MIN_HAMMING:-0.2}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

INITIAL_BUFFER_ARGS=(--initial_buffer_mode "${INITIAL_BUFFER_MODE}")
INITIAL_BUFFER_NAME="${INITIAL_BUFFER_MODE}"
if [ "${INITIAL_BUFFER_MODE}" = "random_blobs" ]; then
  INITIAL_BUFFER_ARGS+=(
    --initial_blob_count_min "${INITIAL_BLOB_COUNT_MIN}"
    --initial_blob_count_max "${INITIAL_BLOB_COUNT_MAX}"
    --initial_blob_radius_min "${INITIAL_BLOB_RADIUS_MIN}"
    --initial_blob_radius_max "${INITIAL_BLOB_RADIUS_MAX}"
    --initial_blob_min_hamming "${INITIAL_BLOB_MIN_HAMMING}"
  )
  INITIAL_BUFFER_NAME="blobinit_b${INITIAL_BLOB_COUNT_MIN}-${INITIAL_BLOB_COUNT_MAX}_r${INITIAL_BLOB_RADIUS_MIN}-${INITIAL_BLOB_RADIUS_MAX}_h${INITIAL_BLOB_MIN_HAMMING}"
fi

COMMON_ARGS=(
  --grid_width 40
  --grid_height 20
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding sorted_material
  --sorted_material_profile binary
  --density_filter_radius 0
  --projection_beta 0
  --generator_type conv
  --generator_output_norm centered_l2
  --generator_channels 64
  --discriminator_type mlp
  --discriminator_hidden_dims 128 128
  --optimizer_type "${GAN_OPTIMIZER_TYPE}"
  --discriminator_steps 1
  --curiosity "${CURIO}"
  --curiosity_space raw
  --curiosity_reference buffer
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --fem_workers "${FEM_WORKERS}"
  --history_interval 25
  --seed "${SEED}"
  "${INITIAL_BUFFER_ARGS[@]}"
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
    --output_dir "${OUT_ROOT}/topopt_sorted_binary_${GAN_OPTIMIZER_TYPE}_gan_${INITIAL_BUFFER_NAME}_bs${BATCH_SIZE}_buf$((BATCH_SIZE * BUFFER_MULTIPLIER))_lr_${label}_iter${ITER}_seed${SEED}"
done
