#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/topopt_logs

ITER="${N_ITER:-3000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
CURIOSITIES="${CURIOSITIES:-0.005 0.01 0.03 1 10 100}"

BASE=(
  python examples/topology_optimization/cantilever_fem.py
  --grid_width 40
  --grid_height 20
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding topk_volume
  --optimizer_type quantile_ranked_default
  --ranker_list_size 64
  --ranker_steps 1
  --ranker_weight 1.0
  --ranker_target_curve exp
  --ranker_tau 8
  --curiosity_space topology
  --curiosity_schedule warmup_cosine
  --curiosity_warmup_frac 0.05
  --curiosity_min 0
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

for curiosity in ${CURIOSITIES}; do
  curiosity_tag="${curiosity//./p}"
  run_name="fem_cantilever_topkvolume_quantile_ranked_default_exp8_w1_iter${ITER}_convG_mlpD_topocuriosity${curiosity_tag}_sched_seed${SEED}"
  out_dir="results/${run_name}"
  log_file="results/topopt_logs/${run_name}.log"

  echo "[$(date)] starting ${run_name}"
  "${BASE[@]}" \
    --curiosity "${curiosity}" \
    --output_dir "${out_dir}" \
    2>&1 | tee "${log_file}"
  echo "[$(date)] finished ${run_name}"
done
