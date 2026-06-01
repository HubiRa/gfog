#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
G_OPTIMIZER="${G_OPTIMIZER:-muon}"
D_OPTIMIZER="${D_OPTIMIZER:-muon}"
G_LR="${G_LR:-0.03}"
D_LR="${D_LR:-0.1}"
SEEDS="${SEEDS:-0}"
FUNCTIONS="${FUNCTIONS:-ackley himmelblau mishra rosenbrock}"
OPTIMIZERS="${OPTIMIZERS:-quantile hybrid}"
OUT_ROOT="${OUT_ROOT:-results/testfunctions_ranker_objectives_suite}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

for seed in ${SEEDS}; do
  for function_name in ${FUNCTIONS}; do
    for optimizer_name in ${OPTIMIZERS}; do
      python examples/testfunctions/compare_ranker_objectives.py \
        --function "${function_name}" \
        --optimizer "${optimizer_name}" \
        --n_iter "${ITER}" \
        --batch_size "${BATCH_SIZE}" \
        --buffer_multiplier "${BUFFER_MULTIPLIER}" \
        --g_optimizer "${G_OPTIMIZER}" \
        --d_optimizer "${D_OPTIMIZER}" \
        --g_lr "${G_LR}" \
        --d_lr "${D_LR}" \
        --ranker_list_size "${BATCH_SIZE}" \
        --ranker_sample_pool_size "$((BATCH_SIZE * BUFFER_MULTIPLIER))" \
        --ranker_target_curve exp \
        --ranker_tau 4 \
        --history_interval 10 \
        --seed "${seed}" \
        --output_dir "${OUT_ROOT}"
    done
  done
done
