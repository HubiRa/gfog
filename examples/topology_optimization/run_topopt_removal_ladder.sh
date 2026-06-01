#!/usr/bin/env bash
set -euo pipefail

# Staged material-removal objective. G emits an importance/priority field; f
# keeps the top-k cells at each volume fraction and returns lexicographic
# compliance-threshold violations plus final low-volume compliance.
#
# Usage:
#   bash examples/topology_optimization/run_topopt_removal_ladder.sh
#   N_ITER=1000 G_LR=0.01 D_LR=0.1 bash examples/topology_optimization/run_topopt_removal_ladder.sh

N_ITER="${N_ITER:-500}"
SEEDS="${SEEDS:-0}"
GRID_WIDTH="${GRID_WIDTH:-40}"
GRID_HEIGHT="${GRID_HEIGHT:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-4}"
LATENT_DIM="${LATENT_DIM:-64}"
FEM_WORKERS="${FEM_WORKERS:-8}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_removal_ladder}"

REMOVAL_VOLUMES="${REMOVAL_VOLUMES:-0.50 0.45 0.40 0.35 0.30}"
REMOVAL_COMPLIANCES="${REMOVAL_COMPLIANCES:-80 95 115 145 190}"
REMOVAL_CONNECTIVITY_MAX="${REMOVAL_CONNECTIVITY_MAX:-}"
REMOVAL_CONNECT_REPAIR="${REMOVAL_CONNECT_REPAIR:-false}"
FINAL_VOLUME="${FINAL_VOLUME:-0.30}"
LOAD_CASE="${LOAD_CASE:-center_point}"

G_LR="${G_LR:-0.01}"
D_LR="${D_LR:-0.1}"
CURIOSITY="${CURIOSITY:-0.0003}"
RANKER_TAU="${RANKER_TAU:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-64}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-128}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"

mkdir -p "${OUTPUT_ROOT}/logs"

DISCRIMINATOR_SPECTRAL_ARGS=()
DISCRIMINATOR_SPECTRAL_NAME=""
if [ "${DISCRIMINATOR_SPECTRAL_NORM}" != "true" ]; then
  DISCRIMINATOR_SPECTRAL_ARGS+=("--no-discriminator_spectral_norm")
  DISCRIMINATOR_SPECTRAL_NAME="_dsn0"
fi

REMOVAL_NAME="$(echo "${REMOVAL_VOLUMES}" | tr ' ' '-')"
REMOVAL_CONNECTIVITY_ARGS=()
REMOVAL_CONNECTIVITY_NAME=""
if [ -n "${REMOVAL_CONNECTIVITY_MAX}" ]; then
  REMOVAL_CONNECTIVITY_ARGS+=("--removal_ladder_connectivity_max" "${REMOVAL_CONNECTIVITY_MAX}")
  REMOVAL_CONNECTIVITY_NAME="_conn${REMOVAL_CONNECTIVITY_MAX}"
fi
if [ "${REMOVAL_CONNECT_REPAIR}" = "true" ]; then
  REMOVAL_CONNECTIVITY_ARGS+=("--binhead_connect_support")
  REMOVAL_CONNECTIVITY_NAME="${REMOVAL_CONNECTIVITY_NAME}_repair"
fi

for seed in ${SEEDS}; do
  run_name="removal_${LOAD_CASE}_vols${REMOVAL_NAME}${REMOVAL_CONNECTIVITY_NAME}_iter${N_ITER}_bs${BATCH_SIZE}_bufx${BUFFER_MULTIPLIER}_gnorm${GENERATOR_OUTPUT_NORM}${DISCRIMINATOR_SPECTRAL_NAME}_muon_muon_glr${G_LR}_dlr${D_LR}_curio${CURIOSITY}_seed${seed}"
  out_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${OUTPUT_ROOT}/logs/${run_name}.log"
  echo "[$(date)] starting ${run_name}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
    --grid_width "${GRID_WIDTH}" \
    --grid_height "${GRID_HEIGHT}" \
    --volume_max "${FINAL_VOLUME}" \
    --n_iter "${N_ITER}" \
    --batch_size "${BATCH_SIZE}" \
    --buffer_multiplier "${BUFFER_MULTIPLIER}" \
    --latent_dim "${LATENT_DIM}" \
    --encoding sorted_material \
    --sorted_material_profile binary \
    --density_filter_radius 0 \
    --projection_beta 0 \
    --load_case "${LOAD_CASE}" \
    --fem_workers "${FEM_WORKERS}" \
    --seed "${seed}" \
    --g_torch_optimizer muon \
    --d_torch_optimizer muon \
    --g_lr "${G_LR}" \
    --d_lr "${D_LR}" \
    --discriminator_steps 1 \
    --generator_type conv \
    --generator_output_norm "${GENERATOR_OUTPUT_NORM}" \
    --discriminator_type mlp \
    --generator_channels 64 \
    --discriminator_hidden_dims 128 128 \
    ${DISCRIMINATOR_SPECTRAL_ARGS[@]+"${DISCRIMINATOR_SPECTRAL_ARGS[@]}"} \
    --optimizer_type quantile_ranked_default \
    --ranker_weight 1.0 \
    --ranker_target_curve exp \
    --ranker_target_scope local \
    --ranker_tau "${RANKER_TAU}" \
    --ranker_list_size "${RANKER_LIST_SIZE}" \
    --ranker_steps 1 \
    --ranker_sample_pool_size "${RANKER_SAMPLE_POOL_SIZE}" \
    --ranker_sample_mode random_top_pool \
    --curiosity "${CURIOSITY}" \
    --curiosity_space raw \
    --curiosity_reference buffer \
    --removal_ladder_volumes ${REMOVAL_VOLUMES} \
    --removal_ladder_compliances ${REMOVAL_COMPLIANCES} \
    ${REMOVAL_CONNECTIVITY_ARGS[@]+"${REMOVAL_CONNECTIVITY_ARGS[@]}"} \
    --history_interval "${HISTORY_INTERVAL}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
  echo "[$(date)] finished ${run_name}"
done
