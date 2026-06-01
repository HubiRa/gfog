#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
LENGTH="${LENGTH:-256}"
ALPHABET_SIZE="${ALPHABET_SIZE:-8}"
POSITION_MODE="${POSITION_MODE:-anywhere}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
SEEDS="${SEEDS:-0}"
OPTIMIZERS="${OPTIMIZERS:-quantile hybrid}"
OUT_ROOT="${OUT_ROOT:-results/discrete_sequence_ranker_objectives_suite}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

for seed in ${SEEDS}; do
  for optimizer_name in ${OPTIMIZERS}; do
    python examples/discrete_sequence_design/compare_ranker_objectives.py \
      --optimizer "${optimizer_name}" \
      --n_iter "${ITER}" \
      --batch_size "${BATCH_SIZE}" \
      --buffer_multiplier "${BUFFER_MULTIPLIER}" \
	      --length "${LENGTH}" \
	      --alphabet_size "${ALPHABET_SIZE}" \
	      --position_mode "${POSITION_MODE}" \
	      --generator_output_norm "${GENERATOR_OUTPUT_NORM}" \
	      --ranker_list_size "${BATCH_SIZE}" \
      --ranker_sample_pool_size "$((BATCH_SIZE * BUFFER_MULTIPLIER))" \
      --ranker_target_curve exp \
      --ranker_tau 4 \
      --seed "${seed}" \
      --output_dir "${OUT_ROOT}"
  done
done
