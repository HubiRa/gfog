#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/topopt_logs

ITER="${N_ITER:-1000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
WEIGHTS="${RANKER_WEIGHTS:-0.01 0.03 0.1 0.3}"

BASE=(
  python examples/topology_optimization/cantilever_fem.py
  --grid_width 40
  --grid_height 20
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding topk_volume
  --optimizer_type ranked_lsgan
  --ranker_list_size 64
  --ranker_steps 1
  --curiosity 0
  --density_filter_radius 1
  --projection_beta 1
  --seed "${SEED}"
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --g_lr 0.03
  --d_lr 0.03
  --generator_type conv
  --discriminator_type mlp
  --generator_channels 64
  --discriminator_hidden_dims 128 128
)

for weight in ${WEIGHTS}; do
  weight_tag="${weight//./p}"
  run_name="fem_cantilever_topkvolume_rankedlsgan_aux${weight_tag}_iter${ITER}_convG_mlpD_seed${SEED}"
  out_dir="results/${run_name}"
  log_file="results/topopt_logs/${run_name}.log"

  echo "[$(date)] starting ${run_name}"
  "${BASE[@]}" \
    --ranker_weight "${weight}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
  echo "[$(date)] finished ${run_name}"
done
