#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/topopt_logs

ITER="${N_ITER:-3000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
CURIO="${CURIO:-0.1}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"

RUN_NAME="fem_cantilever_topkvolume_lsgan_iter${ITER}_convG_mlpD_topocuriosity${CURIO}_sched_batch${BATCH_SIZE}_bufx${BUFFER_MULTIPLIER}_seed${SEED}"
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
  --encoding topk_volume \
  --optimizer_type lsgan \
  --curiosity "${CURIO}" \
  --curiosity_space topology \
  --curiosity_schedule warmup_cosine \
  --curiosity_warmup_frac "${WARMUP_FRAC}" \
  --curiosity_min 0 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --seed "${SEED}" \
  --g_torch_optimizer muon \
  --d_torch_optimizer muon \
  --g_lr 0.03 \
  --d_lr 0.03 \
  --generator_type conv \
  --discriminator_type mlp \
  --generator_channels 64 \
  --discriminator_hidden_dims 128 128 \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

echo "[$(date)] finished ${RUN_NAME}"
