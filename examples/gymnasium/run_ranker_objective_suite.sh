#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
G_OPTIMIZER="${G_OPTIMIZER:-muon}"
D_OPTIMIZER="${D_OPTIMIZER:-muon}"
G_LR="${G_LR:-0.03}"
D_LR="${D_LR:-0.1}"
RUNS_PER_ENV="${RUNS_PER_ENV:-5}"
EPISODE_STEPS="${EPISODE_STEPS:-500}"
SEEDS="${SEEDS:-0}"
OPTIMIZERS="${OPTIMIZERS:-quantile hybrid}"
OUT_ROOT="${OUT_ROOT:-results/gymnasium_ranker_objectives_suite}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

for seed in ${SEEDS}; do
  for optimizer_name in ${OPTIMIZERS}; do
    python examples/gymnasium/compare_ranker_objectives.py \
      --task cartpole \
      --optimizer "${optimizer_name}" \
      --n_iter "${ITER}" \
      --batch_size "${BATCH_SIZE}" \
      --buffer_multiplier "${BUFFER_MULTIPLIER}" \
      --g_optimizer "${G_OPTIMIZER}" \
      --d_optimizer "${D_OPTIMIZER}" \
      --g_lr "${G_LR}" \
      --d_lr "${D_LR}" \
      --runs_per_env "${RUNS_PER_ENV}" \
      --episode_steps "${EPISODE_STEPS}" \
      --ranker_list_size "${BATCH_SIZE}" \
      --ranker_sample_pool_size "$((BATCH_SIZE * BUFFER_MULTIPLIER))" \
      --ranker_target_curve exp \
      --ranker_tau 4 \
      --history_interval 5 \
      --seed "${seed}" \
      --output_dir "${OUT_ROOT}"
  done
done
