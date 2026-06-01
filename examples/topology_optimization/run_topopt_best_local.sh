#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/topopt_logs

ITER="${N_ITER:-3000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-8}"
CURIO="${CURIO:-0.0003}"
FEM_WORKERS="${FEM_WORKERS:-8}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"
LATENT_DISTRIBUTION="${LATENT_DISTRIBUTION:-normal}"
LATENT_UNIFORM_LOW="${LATENT_UNIFORM_LOW:--1}"
LATENT_UNIFORM_HIGH="${LATENT_UNIFORM_HIGH:-1}"
FIXED_LATENT_BANK="${FIXED_LATENT_BANK:-false}"
FIXED_LATENT_BANK_SIZE="${FIXED_LATENT_BANK_SIZE:-}"
FIXED_LATENT_SELECTION="${FIXED_LATENT_SELECTION:-output_diverse}"
FIXED_LATENT_CANDIDATE_MULTIPLIER="${FIXED_LATENT_CANDIDATE_MULTIPLIER:-8}"
FIXED_LATENT_SAMPLE_MODE="${FIXED_LATENT_SAMPLE_MODE:-shuffle_cycle}"
FIXED_LATENT_CHUNK_SIZE="${FIXED_LATENT_CHUNK_SIZE:-1024}"
FIXED_LATENT_NOISE_STD="${FIXED_LATENT_NOISE_STD:-0}"
FIXED_LATENT_NOISE_NORMALIZE="${FIXED_LATENT_NOISE_NORMALIZE:-true}"
FIXED_LATENT_UNIFORMITY_WEIGHT="${FIXED_LATENT_UNIFORMITY_WEIGHT:-0}"
FIXED_LATENT_UNIFORMITY_BATCH_SIZE="${FIXED_LATENT_UNIFORMITY_BATCH_SIZE:-128}"
FIXED_LATENT_UNIFORMITY_SAMPLE_MODE="${FIXED_LATENT_UNIFORMITY_SAMPLE_MODE:-shuffle_cycle}"
FIXED_LATENT_UNIFORMITY_T="${FIXED_LATENT_UNIFORMITY_T:-2}"

FIXED_LATENT_ARGS=()
FIXED_LATENT_NAME="freshz"
if [ "${FIXED_LATENT_BANK}" = "true" ]; then
  FIXED_LATENT_NAME="fixedz_${FIXED_LATENT_SELECTION}_x${FIXED_LATENT_CANDIDATE_MULTIPLIER}_${FIXED_LATENT_SAMPLE_MODE}"
  FIXED_LATENT_ARGS+=("--fixed_latent_bank")
  FIXED_LATENT_ARGS+=("--fixed_latent_selection" "${FIXED_LATENT_SELECTION}")
  FIXED_LATENT_ARGS+=("--fixed_latent_candidate_multiplier" "${FIXED_LATENT_CANDIDATE_MULTIPLIER}")
  FIXED_LATENT_ARGS+=("--fixed_latent_sample_mode" "${FIXED_LATENT_SAMPLE_MODE}")
  FIXED_LATENT_ARGS+=("--fixed_latent_chunk_size" "${FIXED_LATENT_CHUNK_SIZE}")
  FIXED_LATENT_ARGS+=("--fixed_latent_noise_std" "${FIXED_LATENT_NOISE_STD}")
  if [ "${FIXED_LATENT_NOISE_NORMALIZE}" != "true" ]; then
    FIXED_LATENT_ARGS+=("--fixed_latent_noise_no_normalize")
  fi
  if [ "${FIXED_LATENT_NOISE_STD}" != "0" ]; then
    FIXED_LATENT_NAME="${FIXED_LATENT_NAME}_noise${FIXED_LATENT_NOISE_STD}"
  fi
  FIXED_LATENT_ARGS+=("--fixed_latent_uniformity_weight" "${FIXED_LATENT_UNIFORMITY_WEIGHT}")
  FIXED_LATENT_ARGS+=("--fixed_latent_uniformity_batch_size" "${FIXED_LATENT_UNIFORMITY_BATCH_SIZE}")
  FIXED_LATENT_ARGS+=("--fixed_latent_uniformity_sample_mode" "${FIXED_LATENT_UNIFORMITY_SAMPLE_MODE}")
  FIXED_LATENT_ARGS+=("--fixed_latent_uniformity_t" "${FIXED_LATENT_UNIFORMITY_T}")
  if [ "${FIXED_LATENT_UNIFORMITY_WEIGHT}" != "0" ]; then
    FIXED_LATENT_NAME="${FIXED_LATENT_NAME}_banku${FIXED_LATENT_UNIFORMITY_WEIGHT}_ub${FIXED_LATENT_UNIFORMITY_BATCH_SIZE}"
  fi
  if [ -n "${FIXED_LATENT_BANK_SIZE}" ]; then
    FIXED_LATENT_ARGS+=("--fixed_latent_bank_size" "${FIXED_LATENT_BANK_SIZE}")
    FIXED_LATENT_NAME="${FIXED_LATENT_NAME}_n${FIXED_LATENT_BANK_SIZE}"
  fi
fi

LATENT_NAME="${LATENT_DISTRIBUTION}"
if [ "${LATENT_DISTRIBUTION}" = "uniform" ]; then
  LATENT_NAME="uniform${LATENT_UNIFORM_LOW}_${LATENT_UNIFORM_HIGH}"
fi

DISCRIMINATOR_SPECTRAL_ARGS=()
DISCRIMINATOR_SPECTRAL_NAME=""
if [ "${DISCRIMINATOR_SPECTRAL_NORM}" != "true" ]; then
  DISCRIMINATOR_SPECTRAL_ARGS+=("--no-discriminator_spectral_norm")
  DISCRIMINATOR_SPECTRAL_NAME="_dsn0"
fi

RUN_NAME="fem_cantilever_sorted_binary_quantile_tau4_iter${ITER}_convG_mlpD_${FIXED_LATENT_NAME}_z${LATENT_NAME}_gnorm${GENERATOR_OUTPUT_NORM}${DISCRIMINATOR_SPECTRAL_NAME}_rawcurio${CURIO}_batch${BATCH_SIZE}_bufx${BUFFER_MULTIPLIER}_seed${SEED}"
OUT_DIR="results/${RUN_NAME}"
LOG_FILE="results/topopt_logs/${RUN_NAME}.log"

echo "[$(date)] starting ${RUN_NAME}"

python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 \
  --grid_height 20 \
  --n_iter "${ITER}" \
  --batch_size "${BATCH_SIZE}" \
  --buffer_multiplier "${BUFFER_MULTIPLIER}" \
  --latent_dim 64 \
  --latent_distribution "${LATENT_DISTRIBUTION}" \
  --latent_uniform_low "${LATENT_UNIFORM_LOW}" \
  --latent_uniform_high "${LATENT_UNIFORM_HIGH}" \
  --encoding sorted_material \
  --sorted_material_profile binary \
  --optimizer_type quantile_ranked_default \
  --ranker_weight 1.0 \
  --ranker_target_curve exp \
  --ranker_target_scope local \
  --ranker_tau 4 \
  --ranker_list_size 64 \
  --ranker_steps 1 \
  --ranker_sample_pool_size 128 \
  --ranker_sample_mode random_top_pool \
  --curiosity "${CURIO}" \
  --curiosity_space raw \
  --curiosity_reference buffer \
  --density_filter_radius 0 \
  --projection_beta 0 \
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
  ${DISCRIMINATOR_SPECTRAL_ARGS[@]+"${DISCRIMINATOR_SPECTRAL_ARGS[@]}"} \
  ${FIXED_LATENT_ARGS[@]+"${FIXED_LATENT_ARGS[@]}"} \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo "[$(date)] finished ${RUN_NAME}"
