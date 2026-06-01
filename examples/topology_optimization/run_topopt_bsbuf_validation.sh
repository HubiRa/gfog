#!/usr/bin/env bash
set -euo pipefail

# Validate the batch-size / buffer-multiplier finding around the current
# sorted-material 40x20 topology baseline.
#
# Defaults compare:
#   - bs128/bufx2: best point from the 1k sweep
#   - bs128/bufx4: nearby larger-buffer control
#   - bs64/bufx8: previous baseline

N_ITER="${N_ITER:-3000}"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-128:2 128:4 64:8}"
FEM_WORKERS="${FEM_WORKERS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_bsbuf_validation_3k}"

for config in ${CONFIGS}; do
  batch_size="${config%%:*}"
  buffer_multiplier="${config##*:}"
  echo "[$(date)] validating batch_size=${batch_size} buffer_multiplier=${buffer_multiplier}"
  env \
    N_ITER="${N_ITER}" \
    SEEDS="${SEEDS}" \
    BATCH_SIZES="${batch_size}" \
    BUFFER_MULTIPLIERS="${buffer_multiplier}" \
    FEM_WORKERS="${FEM_WORKERS}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    bash examples/topology_optimization/run_topopt_ttur_sweep.sh
done
