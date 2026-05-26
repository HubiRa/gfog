#!/usr/bin/env bash
set -euo pipefail

# Diagnostic GFog understanding sweep. This is intentionally not a crypto
# attack benchmark; it compares structured controls against random-oracle-like
# controls under matched evaluation budgets.

ITER="${ITER:-1000}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MODES="${MODES:-byte_linear:0 toy_arx:1 sha256:0}"

BATCHES="${BATCHES:-512}"
BUFFER_MULTIPLIERS="${BUFFER_MULTIPLIERS:-2}"
LATENTS="${LATENTS:-256}"
HIDDENS="${HIDDENS:-256}"

G_LR="${G_LR:-0.003}"
D_LR="${D_LR:-0.03}"
G_OPTIMIZER="${G_OPTIMIZER:-muon}"
D_OPTIMIZER="${D_OPTIMIZER:-muon}"

GENERATOR_OUTPUT_TRANSFORM="${GENERATOR_OUTPUT_TRANSFORM:-softplus_l2}"
GENERATOR_OUTPUT_TEMPERATURE="${GENERATOR_OUTPUT_TEMPERATURE:-10}"
GENERATOR_OUTPUT_BIAS_INIT="${GENERATOR_OUTPUT_BIAS_INIT:--2}"
GENERATOR_OUTPUT_SCALE="${GENERATOR_OUTPUT_SCALE:-1}"

CURIOSITY="${CURIOSITY:-0.3}"
CURIOSITY_REFERENCE="${CURIOSITY_REFERENCE:-batch}"
CURIOSITY_T="${CURIOSITY_T:-2}"

RANKER_TAU="${RANKER_TAU:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-128}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-256}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"
export MPLCONFIGDIR

for mode_round in ${MODES}; do
  mode="${mode_round%%:*}"
  rounds="${mode_round##*:}"
  for batch_size in ${BATCHES}; do
    for buffer_multiplier in ${BUFFER_MULTIPLIERS}; do
      for latent_dim in ${LATENTS}; do
        for hidden_dim in ${HIDDENS}; do
          for seed in ${SEEDS}; do
            output_dir="results/understanding_gfog_crypto_${mode}_r${rounds}_iter${ITER}_bs${batch_size}_bufx${buffer_multiplier}_latent${latent_dim}_hidden${hidden_dim}_glr${G_LR}_dlr${D_LR}_curio${CURIOSITY}_seed${seed}"
            echo "Running ${output_dir}"
            python examples/crypto_pow/sha_pow.py \
              --candidate_encoding binhead_bits \
              --generator_output_transform "${GENERATOR_OUTPUT_TRANSFORM}" \
              --generator_output_temperature "${GENERATOR_OUTPUT_TEMPERATURE}" \
              --generator_output_bias_init "${GENERATOR_OUTPUT_BIAS_INIT}" \
              --generator_output_scale "${GENERATOR_OUTPUT_SCALE}" \
              --curiosity "${CURIOSITY}" \
              --curiosity_reference "${CURIOSITY_REFERENCE}" \
              --curiosity_t "${CURIOSITY_T}" \
              --hash_mode "${mode}" \
              --toy_rounds "${rounds}" \
              --n_iter "${ITER}" \
              --batch_size "${batch_size}" \
              --buffer_multiplier "${buffer_multiplier}" \
              --latent_dim "${latent_dim}" \
              --hidden_dim "${hidden_dim}" \
              --g_optimizer "${G_OPTIMIZER}" \
              --d_optimizer "${D_OPTIMIZER}" \
              --g_lr "${G_LR}" \
              --d_lr "${D_LR}" \
              --ranker_tau "${RANKER_TAU}" \
              --ranker_list_size "${RANKER_LIST_SIZE}" \
              --ranker_sample_pool_size "${RANKER_SAMPLE_POOL_SIZE}" \
              --seed "${seed}" \
              --output_dir "${output_dir}"
          done
        done
      done
    done
  done
done
