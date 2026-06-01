#!/usr/bin/env bash
set -euo pipefail

# Same-grid topology optimization across different right-edge load distributions.
# Keeps the topology representation fixed and varies only f's force vector.
#
# Usage:
#   bash examples/topology_optimization/run_topopt_loadcase_suite.sh
#   N_ITER=3000 FEM_WORKERS=8 LOAD_CASES="center_point right_top_point right_edge_uniform" bash examples/topology_optimization/run_topopt_loadcase_suite.sh

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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_loadcase_suite}"
LOAD_CASES="${LOAD_CASES:-center_point right_top_point right_bottom_point right_two_points right_edge_uniform right_edge_shear}"
SETTING="${SETTING:-quantile_best}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
QUANTILE_CURIOSITY="${QUANTILE_CURIOSITY:-0.0003}"

mkdir -p "${OUTPUT_ROOT}/logs"

run_one() {
  local load_case="$1"
  local out_dir="${OUTPUT_ROOT}/${load_case}_iter${N_ITER}_seed${SEED}"
  local log_file="${OUTPUT_ROOT}/logs/${load_case}_iter${N_ITER}_seed${SEED}.log"
  echo "[$(date)] starting load_case=${load_case}"
  local method_args=()
  case "${SETTING}" in
    quantile_best)
      method_args=(
        --optimizer_type quantile_ranked_default
        --ranker_weight 1.0
        --ranker_target_curve exp
        --ranker_target_scope local
        --ranker_tau "${RANKER_TAU}"
        --curiosity "${QUANTILE_CURIOSITY}"
        --curiosity_space raw
        --curiosity_reference buffer
      )
      ;;
    mixed_ranker_curiosity)
      method_args=(
        --optimizer_type hybrid_contextual_utility
        --d_score_center_weight 0.01
        --d_score_scale_weight 0.01
        --d_score_target_std 1.0
        --utility_target_scale 100.0
        --utility_weight 0.05
        --generator_utility_weight 0.05
        --utility_clip 3.0
        --curiosity 0.003
        --curiosity_space raw
        --curiosity_reference batch
        --curiosity_schedule warmup_cosine_annealing
        --curiosity_cycles 4
        --curiosity_min 0
      )
      ;;
    *)
      echo "Unknown SETTING=${SETTING}. Expected quantile_best or mixed_ranker_curiosity." >&2
      exit 2
      ;;
  esac

  MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
    --grid_width "${GRID_WIDTH}" \
    --grid_height "${GRID_HEIGHT}" \
    --n_iter "${N_ITER}" \
    --batch_size "${BATCH_SIZE}" \
    --buffer_multiplier "${BUFFER_MULTIPLIER}" \
    --latent_dim "${LATENT_DIM}" \
    --encoding sorted_material \
    --sorted_material_profile binary \
    --density_filter_radius 0 \
    --projection_beta 0 \
    --load_case "${load_case}" \
    --fem_workers "${FEM_WORKERS}" \
    --seed "${SEED}" \
    --g_torch_optimizer muon \
    --d_torch_optimizer muon \
    --g_lr 0.03 \
    --d_lr 0.1 \
    --discriminator_steps 1 \
    --generator_type conv \
    --generator_output_norm "${GENERATOR_OUTPUT_NORM}" \
    --discriminator_type mlp \
    --generator_channels 64 \
    --discriminator_hidden_dims 128 128 \
    --ranker_list_size "${RANKER_LIST_SIZE}" \
    --ranker_steps 1 \
    --ranker_sample_pool_size "${RANKER_SAMPLE_POOL_SIZE}" \
    --ranker_sample_mode random_top_pool \
    --history_interval "${HISTORY_INTERVAL}" \
    "${method_args[@]}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
  echo "[$(date)] finished load_case=${load_case}"
}

for load_case in ${LOAD_CASES}; do
  run_one "${load_case}"
done
