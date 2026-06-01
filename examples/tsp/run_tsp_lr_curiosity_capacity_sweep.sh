#!/usr/bin/env bash
set -euo pipefail

# Fast TSP argsort sweep for optimizer TTUR, curiosity, model capacity, and normalization.
#
# Usage:
#   PRESET=compact MAX_JOBS=4 bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh
#   PRESET=full MAX_JOBS=8 N_ITER=2000 bash examples/tsp/run_tsp_lr_curiosity_capacity_sweep.sh

PRESET="${PRESET:-compact}"
MAX_JOBS="${MAX_JOBS:-4}"
N_CITIES="${N_CITIES:-50}"
N_ITER="${N_ITER:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-128}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-512}"
RANKER_TAU="${RANKER_TAU:-4}"
LATENT_DIM="${LATENT_DIM:-64}"
CITY_SEED="${CITY_SEED:-0}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/tsp_argsort_sweep}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
TWO_OPT_STARTS="${TWO_OPT_STARTS:-64}"
OBJECTIVE_TWO_OPT_PASSES="${OBJECTIVE_TWO_OPT_PASSES:-0}"
GENERATOR_TYPE="${GENERATOR_TYPE:-mlp}"
DISCRIMINATOR_TYPE="${DISCRIMINATOR_TYPE:-mlp}"
SET_GENERATOR_DIM="${SET_GENERATOR_DIM:-256}"
SET_GENERATOR_DEPTH="${SET_GENERATOR_DEPTH:-2}"
SET_GENERATOR_HEADS="${SET_GENERATOR_HEADS:-4}"
SET_GENERATOR_MLP_RATIO="${SET_GENERATOR_MLP_RATIO:-2}"
SET_DISCRIMINATOR_DIM="${SET_DISCRIMINATOR_DIM:-256}"
SET_DISCRIMINATOR_DEPTH="${SET_DISCRIMINATOR_DEPTH:-2}"
SET_DISCRIMINATOR_HEADS="${SET_DISCRIMINATOR_HEADS:-4}"
SET_DISCRIMINATOR_MLP_RATIO="${SET_DISCRIMINATOR_MLP_RATIO:-2}"
SET_DROPOUT="${SET_DROPOUT:-0}"
ROUTE_PRIOR="${ROUTE_PRIOR:-none}"
ROUTE_PRIOR_ALPHA="${ROUTE_PRIOR_ALPHA:-1}"
ROUTE_PRIOR_HILBERT_BITS="${ROUTE_PRIOR_HILBERT_BITS:-10}"

case "$PRESET" in
  smoke)
    objectives=("quantile")
    opt_pairs=("muon:muon")
    lr_pairs=("0.03:0.1")
    curiosities=("0.0003")
    hidden_pairs=("128,128:128,128")
    norms=("centered_l2")
    taus=("$RANKER_TAU")
    batch_configs=("$BATCH_SIZE:$BUFFER_MULTIPLIER:$RANKER_LIST_SIZE:$RANKER_SAMPLE_POOL_SIZE")
    city_seeds=("$CITY_SEED")
    seeds=("$SEED")
    route_priors=("$ROUTE_PRIOR")
    route_prior_alphas=("$ROUTE_PRIOR_ALPHA")
    ;;
  compact)
    objectives=("quantile")
    opt_pairs=("muon:muon" "adamw:adamw")
    lr_pairs=("0.03:0.1" "0.01:0.03" "0.003:0.03" "0.001:0.01")
    curiosities=("0" "0.0003" "0.003" "0.03")
    hidden_pairs=("128,128:128,128" "256,256:256,256")
    norms=("centered_l2")
    taus=("$RANKER_TAU")
    batch_configs=("$BATCH_SIZE:$BUFFER_MULTIPLIER:$RANKER_LIST_SIZE:$RANKER_SAMPLE_POOL_SIZE")
    city_seeds=("$CITY_SEED")
    seeds=("$SEED")
    route_priors=("$ROUTE_PRIOR")
    route_prior_alphas=("$ROUTE_PRIOR_ALPHA")
    ;;
  full)
    objectives=("quantile" "hybrid")
    opt_pairs=("muon:muon" "adam:adam" "adamw:adamw")
    lr_pairs=("0.03:0.1" "0.01:0.03" "0.003:0.03" "0.001:0.01" "0.0003:0.003")
    curiosities=("0" "0.0003" "0.001" "0.003" "0.03" "0.1" "0.3")
    hidden_pairs=("128,128:128,128" "256,256:256,256" "256,256,256:256,256,256")
    norms=("centered_l2" "layernorm" "none")
    taus=("4" "8" "16")
    batch_configs=("$BATCH_SIZE:$BUFFER_MULTIPLIER:$RANKER_LIST_SIZE:$RANKER_SAMPLE_POOL_SIZE")
    city_seeds=("$CITY_SEED")
    seeds=("$SEED")
    route_priors=("$ROUTE_PRIOR")
    route_prior_alphas=("$ROUTE_PRIOR_ALPHA")
    ;;
  *)
    echo "Unknown PRESET=$PRESET. Expected smoke, compact, or full." >&2
    exit 2
    ;;
esac

if [ -n "${OBJECTIVES:-}" ]; then
  read -r -a objectives <<< "$OBJECTIVES"
fi
if [ -n "${OPT_PAIRS:-}" ]; then
  read -r -a opt_pairs <<< "$OPT_PAIRS"
fi
if [ -n "${LR_PAIRS:-}" ]; then
  read -r -a lr_pairs <<< "$LR_PAIRS"
fi
if [ -n "${CURIOSITIES:-}" ]; then
  read -r -a curiosities <<< "$CURIOSITIES"
fi
if [ -n "${HIDDEN_PAIRS:-}" ]; then
  read -r -a hidden_pairs <<< "$HIDDEN_PAIRS"
fi
if [ -n "${NORMS:-}" ]; then
  read -r -a norms <<< "$NORMS"
fi
if [ -n "${TAUS:-}" ]; then
  read -r -a taus <<< "$TAUS"
fi
if [ -n "${BATCH_CONFIGS:-}" ]; then
  read -r -a batch_configs <<< "$BATCH_CONFIGS"
fi
if [ -n "${CITY_SEEDS:-}" ]; then
  read -r -a city_seeds <<< "$CITY_SEEDS"
fi
if [ -n "${SEEDS:-}" ]; then
  read -r -a seeds <<< "$SEEDS"
fi
if [ -n "${ROUTE_PRIORS:-}" ]; then
  read -r -a route_priors <<< "$ROUTE_PRIORS"
fi
if [ -n "${ROUTE_PRIOR_ALPHAS:-}" ]; then
  read -r -a route_prior_alphas <<< "$ROUTE_PRIOR_ALPHAS"
fi

mkdir -p "$OUTPUT_DIR"

wait_for_slot() {
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
    sleep 1
  done
}

run_one() {
  local objective="$1"
  local opt_pair="$2"
  local lr_pair="$3"
  local curiosity="$4"
  local hidden_pair="$5"
  local norm="$6"
  local tau="$7"
  local batch_config="$8"
  local city_seed="$9"
  local seed="${10}"
  local route_prior="${11}"
  local route_prior_alpha="${12}"

  local g_optimizer="${opt_pair%%:*}"
  local d_optimizer="${opt_pair##*:}"
  local g_lr="${lr_pair%%:*}"
  local d_lr="${lr_pair##*:}"
  local g_hidden="${hidden_pair%%:*}"
  local d_hidden="${hidden_pair##*:}"
  local batch_size="${batch_config%%:*}"
  local rest="${batch_config#*:}"
  local buffer_multiplier="${rest%%:*}"
  rest="${rest#*:}"
  local ranker_list_size="${rest%%:*}"
  local ranker_sample_pool_size="${rest##*:}"

  MPLCONFIGDIR=/private/tmp/fontcache python examples/tsp/tsp_argsort.py \
    --n_cities "$N_CITIES" \
    --city_seed "$city_seed" \
    --optimizer "$objective" \
    --n_iter "$N_ITER" \
    --batch_size "$batch_size" \
    --buffer_multiplier "$buffer_multiplier" \
    --latent_dim "$LATENT_DIM" \
    --generator_type "$GENERATOR_TYPE" \
    --generator_hidden_dims "$g_hidden" \
    --discriminator_type "$DISCRIMINATOR_TYPE" \
    --discriminator_hidden_dims "$d_hidden" \
    --set_generator_dim "$SET_GENERATOR_DIM" \
    --set_generator_depth "$SET_GENERATOR_DEPTH" \
    --set_generator_heads "$SET_GENERATOR_HEADS" \
    --set_generator_mlp_ratio "$SET_GENERATOR_MLP_RATIO" \
    --set_discriminator_dim "$SET_DISCRIMINATOR_DIM" \
    --set_discriminator_depth "$SET_DISCRIMINATOR_DEPTH" \
    --set_discriminator_heads "$SET_DISCRIMINATOR_HEADS" \
    --set_discriminator_mlp_ratio "$SET_DISCRIMINATOR_MLP_RATIO" \
    --set_dropout "$SET_DROPOUT" \
    --generator_output_norm "$norm" \
    --route_prior "$route_prior" \
    --route_prior_alpha "$route_prior_alpha" \
    --route_prior_hilbert_bits "$ROUTE_PRIOR_HILBERT_BITS" \
    --g_optimizer "$g_optimizer" \
    --d_optimizer "$d_optimizer" \
    --g_lr "$g_lr" \
    --d_lr "$d_lr" \
    --curiosity "$curiosity" \
    --curiosity_reference batch \
    --ranker_list_size "$ranker_list_size" \
    --ranker_sample_pool_size "$ranker_sample_pool_size" \
    --ranker_tau "$tau" \
    --objective_two_opt_passes "$OBJECTIVE_TWO_OPT_PASSES" \
    --two_opt_starts "$TWO_OPT_STARTS" \
    --history_interval "$HISTORY_INTERVAL" \
    --seed "$seed" \
    --output_dir "$OUTPUT_DIR"
}

total=0
for objective in "${objectives[@]}"; do
  for opt_pair in "${opt_pairs[@]}"; do
    for lr_pair in "${lr_pairs[@]}"; do
      for curiosity in "${curiosities[@]}"; do
        for hidden_pair in "${hidden_pairs[@]}"; do
          for norm in "${norms[@]}"; do
            for tau in "${taus[@]}"; do
              for batch_config in "${batch_configs[@]}"; do
                for city_seed in "${city_seeds[@]}"; do
                  for seed in "${seeds[@]}"; do
                    for route_prior in "${route_priors[@]}"; do
                      for route_prior_alpha in "${route_prior_alphas[@]}"; do
                        total=$((total + 1))
                        wait_for_slot
                        echo "[$total] $objective opt=$opt_pair lr=$lr_pair curio=$curiosity hidden=$hidden_pair norm=$norm tau=$tau batch=$batch_config city_seed=$city_seed seed=$seed prior=$route_prior alpha=$route_prior_alpha"
                        run_one "$objective" "$opt_pair" "$lr_pair" "$curiosity" "$hidden_pair" "$norm" "$tau" "$batch_config" "$city_seed" "$seed" "$route_prior" "$route_prior_alpha" &
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

wait

python examples/tsp/summarize_tsp_results.py \
  --results_dir "$OUTPUT_DIR" \
  --top 25 \
  --csv "$OUTPUT_DIR/summary.csv"
