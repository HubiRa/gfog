#!/usr/bin/env bash
set -euo pipefail

# Depth-8 set/set topology sweep.
# Keeps the current sorted-material quantile-ranker baseline fixed, but swaps to
# set_conv G + set_transformer D and sweeps TTUR learning rates.

N_ITER="${N_ITER:-1000}"
SEEDS="${SEEDS:-0}"
G_LRS="${G_LRS:-0.003 0.01 0.03}"
D_LRS="${D_LRS:-0.03 0.1}"

GRID_WIDTH="${GRID_WIDTH:-40}"
GRID_HEIGHT="${GRID_HEIGHT:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-8}"
LATENT_DIM="${LATENT_DIM:-64}"
FEM_WORKERS="${FEM_WORKERS:-8}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_setset_depth8_lr_sweep}"

RANKER_TAU="${RANKER_TAU:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-64}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-128}"

GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
GENERATOR_CHANNELS="${GENERATOR_CHANNELS:-64}"
SET_GENERATOR_DIM="${SET_GENERATOR_DIM:-128}"
SET_GENERATOR_DEPTH="${SET_GENERATOR_DEPTH:-8}"
SET_GENERATOR_HEADS="${SET_GENERATOR_HEADS:-4}"
SET_DISCRIMINATOR_DIM="${SET_DISCRIMINATOR_DIM:-128}"
SET_DISCRIMINATOR_DEPTH="${SET_DISCRIMINATOR_DEPTH:-8}"
SET_DISCRIMINATOR_HEADS="${SET_DISCRIMINATOR_HEADS:-4}"

CURIOSITY="${CURIOSITY:-0.0003}"
CURIOSITY_SPACE="${CURIOSITY_SPACE:-raw}"
CURIOSITY_REFERENCE="${CURIOSITY_REFERENCE:-buffer}"

mkdir -p "${OUTPUT_ROOT}/logs"

for seed in ${SEEDS}; do
  for g_lr in ${G_LRS}; do
    for d_lr in ${D_LRS}; do
      run_name="setconv_setD_depth${SET_GENERATOR_DEPTH}x${SET_DISCRIMINATOR_DEPTH}_tau${RANKER_TAU}_iter${N_ITER}_bs${BATCH_SIZE}_bufx${BUFFER_MULTIPLIER}_gnorm${GENERATOR_OUTPUT_NORM}_glr${g_lr}_dlr${d_lr}_curio${CURIOSITY}_${CURIOSITY_SPACE}_${CURIOSITY_REFERENCE}_seed${seed}"
      out_dir="${OUTPUT_ROOT}/${run_name}"
      log_file="${OUTPUT_ROOT}/logs/${run_name}.log"
      echo "[$(date)] starting ${run_name}"
      MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
        --grid_width "${GRID_WIDTH}" \
        --grid_height "${GRID_HEIGHT}" \
        --n_iter "${N_ITER}" \
        --batch_size "${BATCH_SIZE}" \
        --buffer_multiplier "${BUFFER_MULTIPLIER}" \
        --buffer_diversity_min_hamming 0 \
        --buffer_diversity_topk_frac 0.48 \
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
        --generator_type set_conv \
        --generator_output_norm "${GENERATOR_OUTPUT_NORM}" \
        --generator_channels "${GENERATOR_CHANNELS}" \
        --set_generator_dim "${SET_GENERATOR_DIM}" \
        --set_generator_depth "${SET_GENERATOR_DEPTH}" \
        --set_generator_heads "${SET_GENERATOR_HEADS}" \
        --discriminator_type set_transformer \
        --set_discriminator_dim "${SET_DISCRIMINATOR_DIM}" \
        --set_discriminator_depth "${SET_DISCRIMINATOR_DEPTH}" \
        --set_discriminator_heads "${SET_DISCRIMINATOR_HEADS}" \
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
        --curiosity_space "${CURIOSITY_SPACE}" \
        --curiosity_reference "${CURIOSITY_REFERENCE}" \
        --curiosity_schedule none \
        --history_interval "${HISTORY_INTERVAL}" \
        --output_dir "${out_dir}" \
        2>&1 | tee "${log_file}"
      echo "[$(date)] finished ${run_name}"
    done
  done
done
