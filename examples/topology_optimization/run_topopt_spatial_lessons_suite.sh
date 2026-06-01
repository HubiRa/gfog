#!/usr/bin/env bash
set -euo pipefail

# Topology experiments that encode the current lessons:
# - keep the spatial Conv-G prior;
# - use centered-L2 generator output normalization for sorted-material genomes;
# - use one ranker/D update per iteration and tune learning rates instead;
# - compare the old quantile fake-rejection objective with mixed evaluated ranking;
# - optionally test batch-only curiosity and connectivity-aware f variants.
#
# Usage:
#   bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh
#   N_ITER=3000 FEM_WORKERS=8 EXPERIMENTS="quantile_best mixed_ranker" bash examples/topology_optimization/run_topopt_spatial_lessons_suite.sh

N_ITER="${N_ITER:-1000}"
SEED="${SEED:-0}"
GRID_WIDTH="${GRID_WIDTH:-40}"
GRID_HEIGHT="${GRID_HEIGHT:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-8}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-64}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-128}"
RANKER_TAU="${RANKER_TAU:-4}"
LATENT_DIM="${LATENT_DIM:-64}"
FEM_WORKERS="${FEM_WORKERS:-1}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_spatial_lessons}"
ENCODING="${ENCODING:-sorted_material}"
SORTED_PROFILE="${SORTED_PROFILE:-binary}"
DENSITY_FILTER_RADIUS="${DENSITY_FILTER_RADIUS:-0}"
PROJECTION_BETA="${PROJECTION_BETA:-0}"
GENERATOR_TYPE="${GENERATOR_TYPE:-conv}"
DISCRIMINATOR_TYPE="${DISCRIMINATOR_TYPE:-mlp}"
GENERATOR_CHANNELS="${GENERATOR_CHANNELS:-64}"
DISCRIMINATOR_HIDDEN_DIMS="${DISCRIMINATOR_HIDDEN_DIMS:-128 128}"
G_LR="${G_LR:-0.03}"
D_LR="${D_LR:-0.1}"
CURIOSITY_SPACE="${CURIOSITY_SPACE:-raw}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
QUANTILE_CURIOSITY="${QUANTILE_CURIOSITY:-0.0003}"
EXPERIMENTS="${EXPERIMENTS:-quantile_best}"

mkdir -p "${OUTPUT_ROOT}/logs"

common_args=(
  --grid_width "${GRID_WIDTH}"
  --grid_height "${GRID_HEIGHT}"
  --n_iter "${N_ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim "${LATENT_DIM}"
  --encoding "${ENCODING}"
  --sorted_material_profile "${SORTED_PROFILE}"
  --density_filter_radius "${DENSITY_FILTER_RADIUS}"
  --projection_beta "${PROJECTION_BETA}"
  --fem_workers "${FEM_WORKERS}"
  --seed "${SEED}"
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --g_lr "${G_LR}"
  --d_lr "${D_LR}"
  --discriminator_steps 1
  --generator_type "${GENERATOR_TYPE}"
  --generator_output_norm "${GENERATOR_OUTPUT_NORM}"
  --discriminator_type "${DISCRIMINATOR_TYPE}"
  --generator_channels "${GENERATOR_CHANNELS}"
  --discriminator_hidden_dims ${DISCRIMINATOR_HIDDEN_DIMS}
  --ranker_list_size "${RANKER_LIST_SIZE}"
  --ranker_steps 1
  --ranker_sample_pool_size "${RANKER_SAMPLE_POOL_SIZE}"
  --ranker_sample_mode random_top_pool
  --history_interval "${HISTORY_INTERVAL}"
)

run_one() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_ROOT}/${name}_iter${N_ITER}_seed${SEED}"
  local log_file="${OUTPUT_ROOT}/logs/${name}_iter${N_ITER}_seed${SEED}.log"
  echo "[$(date)] starting ${name}"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
    "${common_args[@]}" \
    "$@" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
  echo "[$(date)] finished ${name}"
}

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    quantile_best)
      run_one "${experiment}" \
        --optimizer_type quantile_ranked_default \
        --ranker_weight 1.0 \
        --ranker_target_curve exp \
        --ranker_target_scope local \
        --ranker_tau "${RANKER_TAU}" \
        --curiosity "${QUANTILE_CURIOSITY}" \
        --curiosity_space "${CURIOSITY_SPACE}" \
        --curiosity_reference buffer
      ;;
    mixed_ranker)
      run_one "${experiment}" \
        --optimizer_type hybrid_contextual_utility \
        --d_score_center_weight 0.01 \
        --d_score_scale_weight 0.01 \
        --d_score_target_std 1.0 \
        --utility_target_scale 100.0 \
        --utility_weight 0.05 \
        --generator_utility_weight 0.05 \
        --utility_clip 3.0 \
        --curiosity 0
      ;;
    mixed_ranker_curiosity)
      run_one "${experiment}" \
        --optimizer_type hybrid_contextual_utility \
        --d_score_center_weight 0.01 \
        --d_score_scale_weight 0.01 \
        --d_score_target_std 1.0 \
        --utility_target_scale 100.0 \
        --utility_weight 0.05 \
        --generator_utility_weight 0.05 \
        --utility_clip 3.0 \
        --curiosity 0.003 \
        --curiosity_space "${CURIOSITY_SPACE}" \
        --curiosity_reference batch \
        --curiosity_schedule warmup_cosine_annealing \
        --curiosity_cycles 4 \
        --curiosity_min 0
      ;;
    mixed_ranker_connectivity)
      run_one "${experiment}" \
        --optimizer_type hybrid_contextual_utility \
        --d_score_center_weight 0.01 \
        --d_score_scale_weight 0.01 \
        --d_score_target_std 1.0 \
        --utility_target_scale 100.0 \
        --utility_weight 0.05 \
        --generator_utility_weight 0.05 \
        --utility_clip 3.0 \
        --curiosity 0 \
        --connectivity_max 0.0
      ;;
    quantile_connect_repair)
      run_one "${experiment}" \
        --optimizer_type quantile_ranked_default \
        --ranker_weight 1.0 \
        --ranker_target_curve exp \
        --ranker_target_scope local \
        --ranker_tau "${RANKER_TAU}" \
        --curiosity 0 \
        --binhead_connect_support
      ;;
    *)
      echo "Unknown experiment '${experiment}'" >&2
      exit 2
      ;;
  esac
done
