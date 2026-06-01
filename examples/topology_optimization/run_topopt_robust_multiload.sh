#!/usr/bin/env bash
set -euo pipefail

# Robust same-grid topology benchmark: one topology is evaluated under several
# existing cantilever load cases, then ranked by an aggregate compliance.
#
# Usage:
#   bash examples/topology_optimization/run_topopt_robust_multiload.sh
#   N_ITER=3000 SEEDS="0 1 2" ROBUST_LOAD_AGGREGATE=cvar bash examples/topology_optimization/run_topopt_robust_multiload.sh

N_ITER="${N_ITER:-1000}"
SEEDS="${SEEDS:-0}"
GRID_WIDTH="${GRID_WIDTH:-40}"
GRID_HEIGHT="${GRID_HEIGHT:-20}"
VOLUME_MAX="${VOLUME_MAX:-0.35}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-4}"
LATENT_DIM="${LATENT_DIM:-64}"
FEM_WORKERS="${FEM_WORKERS:-8}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_robust_multiload}"

G_LR="${G_LR:-0.07}"
D_LR="${D_LR:-0.08}"
G_LRS="${G_LRS:-${G_LR}}"
D_LRS="${D_LRS:-${D_LR}}"
CURIOSITY="${CURIOSITY:-0.0003}"
RANKER_TAU="${RANKER_TAU:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-64}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-128}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"

ROBUST_LOAD_CASES="${ROBUST_LOAD_CASES:-center_point right_top_point right_bottom_point right_edge_uniform}"
ROBUST_LOAD_AGGREGATE="${ROBUST_LOAD_AGGREGATE:-max}"
ROBUST_LOAD_CVAR_FRAC="${ROBUST_LOAD_CVAR_FRAC:-0.5}"

mkdir -p "${OUTPUT_ROOT}/logs"

DISCRIMINATOR_SPECTRAL_ARGS=()
DISCRIMINATOR_SPECTRAL_NAME=""
if [ "${DISCRIMINATOR_SPECTRAL_NORM}" != "true" ]; then
  DISCRIMINATOR_SPECTRAL_ARGS+=("--no-discriminator_spectral_norm")
  DISCRIMINATOR_SPECTRAL_NAME="_dsn0"
fi

ROBUST_LOAD_NAME="$(echo "${ROBUST_LOAD_CASES}" | tr ' ' '-')"
ROBUST_ARGS=(
  --robust_load_cases ${ROBUST_LOAD_CASES}
  --robust_load_aggregate "${ROBUST_LOAD_AGGREGATE}"
  --robust_load_cvar_frac "${ROBUST_LOAD_CVAR_FRAC}"
)

for seed in ${SEEDS}; do
  for g_lr in ${G_LRS}; do
    for d_lr in ${D_LRS}; do
      run_name="robust_${ROBUST_LOAD_AGGREGATE}_${ROBUST_LOAD_NAME}_iter${N_ITER}_vol${VOLUME_MAX}_bs${BATCH_SIZE}_bufx${BUFFER_MULTIPLIER}_gnorm${GENERATOR_OUTPUT_NORM}${DISCRIMINATOR_SPECTRAL_NAME}_muon_muon_glr${g_lr}_dlr${d_lr}_curio${CURIOSITY}_seed${seed}"
      out_dir="${OUTPUT_ROOT}/${run_name}"
      log_file="${OUTPUT_ROOT}/logs/${run_name}.log"
      echo "[$(date)] starting ${run_name}"
      MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
        --grid_width "${GRID_WIDTH}" \
        --grid_height "${GRID_HEIGHT}" \
        --volume_max "${VOLUME_MAX}" \
        --n_iter "${N_ITER}" \
        --batch_size "${BATCH_SIZE}" \
        --buffer_multiplier "${BUFFER_MULTIPLIER}" \
        --latent_dim "${LATENT_DIM}" \
        --encoding sorted_material \
        --sorted_material_profile binary \
        --density_filter_radius 0 \
        --projection_beta 0 \
        --fem_workers "${FEM_WORKERS}" \
        --seed "${seed}" \
        --g_torch_optimizer muon \
        --d_torch_optimizer muon \
        --g_lr "${g_lr}" \
        --d_lr "${d_lr}" \
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
        --history_interval "${HISTORY_INTERVAL}" \
        "${ROBUST_ARGS[@]}" \
        --output_dir "${out_dir}" \
        2>&1 | tee "${log_file}"
      echo "[$(date)] finished ${run_name}"
    done
  done
done
