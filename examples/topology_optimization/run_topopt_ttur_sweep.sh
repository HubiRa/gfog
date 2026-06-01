#!/usr/bin/env bash
set -euo pipefail

# Current 40x20 sorted-material topology baseline. Defaults run the best
# normalized quantile-ranker setting; override G_LRS/D_LRS/CURIOSITIES for
# sweeps.

N_ITER="${N_ITER:-3000}"
SEEDS="${SEEDS:-0 1 2}"
G_LRS="${G_LRS:-0.03}"
D_LRS="${D_LRS:-0.1}"
OPTIMIZER_PAIRS="${OPTIMIZER_PAIRS:-muon:muon}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-quantile_ranked_default}"

GRID_WIDTH="${GRID_WIDTH:-40}"
GRID_HEIGHT="${GRID_HEIGHT:-20}"
VOLUME_MAX="${VOLUME_MAX:-0.48}"
ENCODING="${ENCODING:-sorted_material}"
COARSE_GRID_WIDTH="${COARSE_GRID_WIDTH:-}"
COARSE_GRID_HEIGHT="${COARSE_GRID_HEIGHT:-}"
SORTED_MATERIAL_PROFILE="${SORTED_MATERIAL_PROFILE:-binary}"
SORTED_MATERIAL_STEEPNESS="${SORTED_MATERIAL_STEEPNESS:-12}"
DENSITY_FILTER_RADIUS="${DENSITY_FILTER_RADIUS:-0}"
PROJECTION_BETA="${PROJECTION_BETA:-0}"
PROJECTION_ETA="${PROJECTION_ETA:-0.5}"
HARD_BINARIZE="${HARD_BINARIZE:-false}"
BAR_COUNT="${BAR_COUNT:-16}"
BAR_WIDTH_MIN="${BAR_WIDTH_MIN:-0.02}"
BAR_WIDTH_MAX="${BAR_WIDTH_MAX:-0.08}"
BAR_EDGE_SOFTNESS="${BAR_EDGE_SOFTNESS:-0.01}"
BATCH_SIZES="${BATCH_SIZES:-${BATCH_SIZE:-64}}"
BUFFER_MULTIPLIERS="${BUFFER_MULTIPLIERS:-${BUFFER_MULTIPLIER:-8}}"
BUFFER_DIVERSITIES="${BUFFER_DIVERSITIES:-0}"
BUFFER_DIVERSITY_TOPK_FRAC="${BUFFER_DIVERSITY_TOPK_FRAC:-0.48}"
LATENT_DIM="${LATENT_DIM:-64}"
LATENT_DISTRIBUTION="${LATENT_DISTRIBUTION:-normal}"
LATENT_UNIFORM_LOW="${LATENT_UNIFORM_LOW:--1}"
LATENT_UNIFORM_HIGH="${LATENT_UNIFORM_HIGH:-1}"
FEM_WORKERS="${FEM_WORKERS:-1}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-25}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/topopt_ttur_sweep}"
RUN_TAG="${RUN_TAG:-}"
VOLUME_LADDER="${VOLUME_LADDER:-}"
COMPLIANCE_LADDER="${COMPLIANCE_LADDER:-}"
ROUGHNESS_LADDER="${ROUGHNESS_LADDER:-}"
CONNECTIVITY_LADDER="${CONNECTIVITY_LADDER:-}"
LEVELS_LADDER="${LEVELS_LADDER:-}"
LEVELS_LADDER_FINAL_OPEN="${LEVELS_LADDER_FINAL_OPEN:-compliance}"

RANKER_TAU="${RANKER_TAU:-4}"
RANKER_LIST_SIZE="${RANKER_LIST_SIZE:-64}"
RANKER_SAMPLE_POOL_SIZE="${RANKER_SAMPLE_POOL_SIZE:-128}"
RANKER_LIST_REPEATS="${RANKER_LIST_REPEATS:-1}"
RANKER_FAKE_WEIGHT="${RANKER_FAKE_WEIGHT:-1}"
RANKER_FAKE_WEIGHTS="${RANKER_FAKE_WEIGHTS:-${RANKER_FAKE_WEIGHT}}"
RANKER_FAKE_REPEATS="${RANKER_FAKE_REPEATS:-1}"
RANKER_FAKE_REPEATS_LIST="${RANKER_FAKE_REPEATS_LIST:-${RANKER_FAKE_REPEATS}}"
UTILITY_TARGET_SCALE="${UTILITY_TARGET_SCALE:-100}"
UTILITY_LOSS="${UTILITY_LOSS:-smooth_l1}"
UTILITY_WEIGHT="${UTILITY_WEIGHT:-0.1}"
UTILITY_WEIGHTS="${UTILITY_WEIGHTS:-${UTILITY_WEIGHT}}"
GENERATOR_UTILITY_WEIGHT="${GENERATOR_UTILITY_WEIGHT:-0}"
UTILITY_CLIP="${UTILITY_CLIP:-3}"
PROPOSAL_POOL_SIZE="${PROPOSAL_POOL_SIZE:-}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-}"
PROPOSAL_DIVERSITY_MIN_HAMMING="${PROPOSAL_DIVERSITY_MIN_HAMMING:-0}"
PROPOSAL_BUFFER_NOVELTY_MIN_HAMMING="${PROPOSAL_BUFFER_NOVELTY_MIN_HAMMING:-0}"
PROPOSAL_BUFFER_NOVELTY_REFERENCE_SIZE="${PROPOSAL_BUFFER_NOVELTY_REFERENCE_SIZE:-128}"
PROPOSAL_BUFFER_REJECT_EXACT_DESIGN_DUPLICATES="${PROPOSAL_BUFFER_REJECT_EXACT_DESIGN_DUPLICATES:-false}"
PROPOSAL_BUFFER_NOVELTY_THRESHOLD="${PROPOSAL_BUFFER_NOVELTY_THRESHOLD:-0.5}"

GENERATOR_CHANNELS="${GENERATOR_CHANNELS:-64}"
GENERATOR_TYPE="${GENERATOR_TYPE:-conv}"
GENERATOR_OUTPUT_NORM="${GENERATOR_OUTPUT_NORM:-centered_l2}"
DISCRIMINATOR_TYPE="${DISCRIMINATOR_TYPE:-mlp}"
DISCRIMINATOR_HIDDEN_DIMS="${DISCRIMINATOR_HIDDEN_DIMS:-128 128}"
DISCRIMINATOR_SPECTRAL_NORM="${DISCRIMINATOR_SPECTRAL_NORM:-true}"

CURIOSITIES="${CURIOSITIES:-0.0003}"
CURIOSITY_SPACE="${CURIOSITY_SPACE:-raw}"
CURIOSITY_REFERENCE="${CURIOSITY_REFERENCE:-buffer}"
CURIOSITY_SCHEDULE="${CURIOSITY_SCHEDULE:-none}"
CURIOSITY_MIN="${CURIOSITY_MIN:-0}"
CURIOSITY_CYCLES="${CURIOSITY_CYCLES:-4}"
PLUMMER_POWER="${PLUMMER_POWER:-1}"
PLUMMER_EPS="${PLUMMER_EPS:-0.001}"
PLUMMER_NORMALIZE="${PLUMMER_NORMALIZE:-layernorm}"
PLUMMER_TERMS="${PLUMMER_TERMS:-batch_buffer}"

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

mkdir -p "${OUTPUT_ROOT}/logs"

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

HARD_BINARIZE_ARGS=()
HARD_BINARIZE_NAME=""
if [ "${HARD_BINARIZE}" = "true" ]; then
  HARD_BINARIZE_ARGS+=("--hard_binarize")
  HARD_BINARIZE_NAME="_hardbin"
fi

COARSE_GRID_ARGS=()
COARSE_GRID_NAME=""
if [ -n "${COARSE_GRID_WIDTH}" ]; then
  COARSE_GRID_ARGS+=("--coarse_grid_width" "${COARSE_GRID_WIDTH}")
  COARSE_GRID_NAME="_cw${COARSE_GRID_WIDTH}"
fi
if [ -n "${COARSE_GRID_HEIGHT}" ]; then
  COARSE_GRID_ARGS+=("--coarse_grid_height" "${COARSE_GRID_HEIGHT}")
  COARSE_GRID_NAME="${COARSE_GRID_NAME}_ch${COARSE_GRID_HEIGHT}"
fi

LADDER_ARGS=()
LADDER_NAME=""
if [ -n "${LEVELS_LADDER}" ]; then
  LADDER_ARGS+=("--levels_ladder")
  for ladder_spec in ${LEVELS_LADDER}; do
    LADDER_ARGS+=("${ladder_spec}")
  done
  LADDER_ARGS+=("--levels_ladder_final_open" "${LEVELS_LADDER_FINAL_OPEN}")
  LADDER_NAME="_levels$(echo "${LEVELS_LADDER}" | tr ' :,' '---')_open${LEVELS_LADDER_FINAL_OPEN}"
else
  if [ -n "${VOLUME_LADDER}" ]; then
    LADDER_ARGS+=("--volume_ladder")
    for ladder_value in ${VOLUME_LADDER}; do
      LADDER_ARGS+=("${ladder_value}")
    done
    LADDER_NAME="${LADDER_NAME}_vlad$(echo "${VOLUME_LADDER}" | tr ' ' '-')"
  fi
  if [ -n "${COMPLIANCE_LADDER}" ]; then
    LADDER_ARGS+=("--compliance_ladder")
    for ladder_value in ${COMPLIANCE_LADDER}; do
      LADDER_ARGS+=("${ladder_value}")
    done
    LADDER_NAME="${LADDER_NAME}_clad$(echo "${COMPLIANCE_LADDER}" | tr ' ' '-')"
  fi
  if [ -n "${ROUGHNESS_LADDER}" ]; then
    LADDER_ARGS+=("--roughness_ladder")
    for ladder_value in ${ROUGHNESS_LADDER}; do
      LADDER_ARGS+=("${ladder_value}")
    done
    LADDER_NAME="${LADDER_NAME}_rlad$(echo "${ROUGHNESS_LADDER}" | tr ' ' '-')"
  fi
  if [ -n "${CONNECTIVITY_LADDER}" ]; then
    LADDER_ARGS+=("--connectivity_ladder")
    for ladder_value in ${CONNECTIVITY_LADDER}; do
      LADDER_ARGS+=("${ladder_value}")
    done
    LADDER_NAME="${LADDER_NAME}_connlad$(echo "${CONNECTIVITY_LADDER}" | tr ' ' '-')"
  fi
fi

if [ -n "${RUN_TAG}" ]; then
  LADDER_NAME="_${RUN_TAG}"
fi

PROPOSAL_ARGS=()
PROPOSAL_NAME=""
if [ -n "${PROPOSAL_POOL_SIZE}" ]; then
  PROPOSAL_ARGS+=("--proposal_pool_size" "${PROPOSAL_POOL_SIZE}")
  PROPOSAL_NAME="${PROPOSAL_NAME}_ppool${PROPOSAL_POOL_SIZE}"
fi
if [ -n "${PROPOSAL_TOP_K}" ]; then
  PROPOSAL_ARGS+=("--proposal_top_k" "${PROPOSAL_TOP_K}")
  PROPOSAL_NAME="${PROPOSAL_NAME}_ptop${PROPOSAL_TOP_K}"
fi
PROPOSAL_ARGS+=("--proposal_diversity_min_hamming" "${PROPOSAL_DIVERSITY_MIN_HAMMING}")
PROPOSAL_ARGS+=("--proposal_buffer_novelty_min_hamming" "${PROPOSAL_BUFFER_NOVELTY_MIN_HAMMING}")
PROPOSAL_ARGS+=("--proposal_buffer_novelty_reference_size" "${PROPOSAL_BUFFER_NOVELTY_REFERENCE_SIZE}")
if [ "${PROPOSAL_BUFFER_REJECT_EXACT_DESIGN_DUPLICATES}" = "true" ]; then
  PROPOSAL_ARGS+=("--proposal_buffer_reject_exact_design_duplicates")
  PROPOSAL_NAME="${PROPOSAL_NAME}_exactdup0_bref${PROPOSAL_BUFFER_NOVELTY_REFERENCE_SIZE}"
fi
PROPOSAL_ARGS+=("--proposal_buffer_novelty_threshold" "${PROPOSAL_BUFFER_NOVELTY_THRESHOLD}")
if [ "${PROPOSAL_DIVERSITY_MIN_HAMMING}" != "0" ]; then
  PROPOSAL_NAME="${PROPOSAL_NAME}_pdiv${PROPOSAL_DIVERSITY_MIN_HAMMING}"
fi
if [ "${PROPOSAL_BUFFER_NOVELTY_MIN_HAMMING}" != "0" ]; then
  PROPOSAL_NAME="${PROPOSAL_NAME}_bnov${PROPOSAL_BUFFER_NOVELTY_MIN_HAMMING}_bref${PROPOSAL_BUFFER_NOVELTY_REFERENCE_SIZE}"
fi

for seed in ${SEEDS}; do
  for batch_size in ${BATCH_SIZES}; do
    for buffer_multiplier in ${BUFFER_MULTIPLIERS}; do
      for optimizer_pair in ${OPTIMIZER_PAIRS}; do
        g_optimizer="${optimizer_pair%%:*}"
        d_optimizer="${optimizer_pair##*:}"
        for g_lr in ${G_LRS}; do
          for d_lr in ${D_LRS}; do
            for curiosity in ${CURIOSITIES}; do
              for buffer_diversity in ${BUFFER_DIVERSITIES}; do
                for utility_weight in ${UTILITY_WEIGHTS}; do
                  for ranker_fake_weight in ${RANKER_FAKE_WEIGHTS}; do
                    for ranker_fake_repeats in ${RANKER_FAKE_REPEATS_LIST}; do
          fixed_latent_args=()
          fixed_latent_name="freshz"
          if [ "${FIXED_LATENT_BANK}" = "true" ]; then
            fixed_latent_name="fixedz_${FIXED_LATENT_SELECTION}_x${FIXED_LATENT_CANDIDATE_MULTIPLIER}_${FIXED_LATENT_SAMPLE_MODE}"
            fixed_latent_args+=("--fixed_latent_bank")
            fixed_latent_args+=("--fixed_latent_selection" "${FIXED_LATENT_SELECTION}")
            fixed_latent_args+=("--fixed_latent_candidate_multiplier" "${FIXED_LATENT_CANDIDATE_MULTIPLIER}")
            fixed_latent_args+=("--fixed_latent_sample_mode" "${FIXED_LATENT_SAMPLE_MODE}")
            fixed_latent_args+=("--fixed_latent_chunk_size" "${FIXED_LATENT_CHUNK_SIZE}")
            fixed_latent_args+=("--fixed_latent_noise_std" "${FIXED_LATENT_NOISE_STD}")
            if [ "${FIXED_LATENT_NOISE_NORMALIZE}" != "true" ]; then
              fixed_latent_args+=("--fixed_latent_noise_no_normalize")
            fi
            if [ "${FIXED_LATENT_NOISE_STD}" != "0" ]; then
              fixed_latent_name="${fixed_latent_name}_noise${FIXED_LATENT_NOISE_STD}"
            fi
            fixed_latent_args+=("--fixed_latent_uniformity_weight" "${FIXED_LATENT_UNIFORMITY_WEIGHT}")
            fixed_latent_args+=("--fixed_latent_uniformity_batch_size" "${FIXED_LATENT_UNIFORMITY_BATCH_SIZE}")
            fixed_latent_args+=("--fixed_latent_uniformity_sample_mode" "${FIXED_LATENT_UNIFORMITY_SAMPLE_MODE}")
            fixed_latent_args+=("--fixed_latent_uniformity_t" "${FIXED_LATENT_UNIFORMITY_T}")
            if [ "${FIXED_LATENT_UNIFORMITY_WEIGHT}" != "0" ]; then
              fixed_latent_name="${fixed_latent_name}_banku${FIXED_LATENT_UNIFORMITY_WEIGHT}_ub${FIXED_LATENT_UNIFORMITY_BATCH_SIZE}"
            fi
            if [ -n "${FIXED_LATENT_BANK_SIZE}" ]; then
              fixed_latent_args+=("--fixed_latent_bank_size" "${FIXED_LATENT_BANK_SIZE}")
              fixed_latent_name="${fixed_latent_name}_n${FIXED_LATENT_BANK_SIZE}"
            fi
          fi
          run_name="${OPTIMIZER_TYPE}_tau${RANKER_TAU}_iter${N_ITER}_${ENCODING}${COARSE_GRID_NAME}_${SORTED_MATERIAL_PROFILE}_vol${VOLUME_MAX}${LADDER_NAME}${HARD_BINARIZE_NAME}_bs${batch_size}_bufx${buffer_multiplier}_bufdiv${buffer_diversity}${PROPOSAL_NAME}_${fixed_latent_name}_z${LATENT_NAME}_${GENERATOR_TYPE}_${DISCRIMINATOR_TYPE}_gnorm${GENERATOR_OUTPUT_NORM}${DISCRIMINATOR_SPECTRAL_NAME}_${g_optimizer}_${d_optimizer}_glr${g_lr}_dlr${d_lr}_curio${curiosity}_${CURIOSITY_SPACE}_${PLUMMER_TERMS}_rrep${RANKER_LIST_REPEATS}_fakew${ranker_fake_weight}_frep${ranker_fake_repeats}_u${utility_weight}_gu${GENERATOR_UTILITY_WEIGHT}_us${UTILITY_TARGET_SCALE}_seed${seed}"
          out_dir="${OUTPUT_ROOT}/${run_name}"
          log_file="${OUTPUT_ROOT}/logs/${run_name}.log"
          echo "[$(date)] starting ${run_name}"
          MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}" python examples/topology_optimization/cantilever_fem.py \
            --grid_width "${GRID_WIDTH}" \
            --grid_height "${GRID_HEIGHT}" \
            ${COARSE_GRID_ARGS[@]+"${COARSE_GRID_ARGS[@]}"} \
            --volume_max "${VOLUME_MAX}" \
            --n_iter "${N_ITER}" \
            --batch_size "${batch_size}" \
            --buffer_multiplier "${buffer_multiplier}" \
            --buffer_diversity_min_hamming "${buffer_diversity}" \
            --buffer_diversity_topk_frac "${BUFFER_DIVERSITY_TOPK_FRAC}" \
            --latent_dim "${LATENT_DIM}" \
            --latent_distribution "${LATENT_DISTRIBUTION}" \
            --latent_uniform_low "${LATENT_UNIFORM_LOW}" \
            --latent_uniform_high "${LATENT_UNIFORM_HIGH}" \
            --encoding "${ENCODING}" \
            --bar_count "${BAR_COUNT}" \
            --bar_width_min "${BAR_WIDTH_MIN}" \
            --bar_width_max "${BAR_WIDTH_MAX}" \
            --bar_edge_softness "${BAR_EDGE_SOFTNESS}" \
            --sorted_material_profile "${SORTED_MATERIAL_PROFILE}" \
            --sorted_material_steepness "${SORTED_MATERIAL_STEEPNESS}" \
            --density_filter_radius "${DENSITY_FILTER_RADIUS}" \
            --projection_beta "${PROJECTION_BETA}" \
            --projection_eta "${PROJECTION_ETA}" \
            ${HARD_BINARIZE_ARGS[@]+"${HARD_BINARIZE_ARGS[@]}"} \
            ${LADDER_ARGS[@]+"${LADDER_ARGS[@]}"} \
            --fem_workers "${FEM_WORKERS}" \
            --seed "${seed}" \
            --g_torch_optimizer "${g_optimizer}" \
            --d_torch_optimizer "${d_optimizer}" \
            --g_lr "${g_lr}" \
            --d_lr "${d_lr}" \
            --discriminator_steps 1 \
            --generator_type "${GENERATOR_TYPE}" \
            --generator_output_norm "${GENERATOR_OUTPUT_NORM}" \
            --discriminator_type "${DISCRIMINATOR_TYPE}" \
            --generator_channels "${GENERATOR_CHANNELS}" \
            --discriminator_hidden_dims ${DISCRIMINATOR_HIDDEN_DIMS} \
            ${DISCRIMINATOR_SPECTRAL_ARGS[@]+"${DISCRIMINATOR_SPECTRAL_ARGS[@]}"} \
            --optimizer_type "${OPTIMIZER_TYPE}" \
            --ranker_weight 1.0 \
            --ranker_target_curve exp \
            --ranker_target_scope local \
            --ranker_tau "${RANKER_TAU}" \
            --ranker_list_size "${RANKER_LIST_SIZE}" \
            --ranker_steps 1 \
            --ranker_list_repeats "${RANKER_LIST_REPEATS}" \
            --ranker_fake_weight "${ranker_fake_weight}" \
            --ranker_fake_repeats "${ranker_fake_repeats}" \
            --ranker_sample_pool_size "${RANKER_SAMPLE_POOL_SIZE}" \
            --ranker_sample_mode random_top_pool \
            ${PROPOSAL_ARGS[@]+"${PROPOSAL_ARGS[@]}"} \
            --utility_target_scale "${UTILITY_TARGET_SCALE}" \
            --utility_loss "${UTILITY_LOSS}" \
            --utility_weight "${utility_weight}" \
            --generator_utility_weight "${GENERATOR_UTILITY_WEIGHT}" \
            --utility_clip "${UTILITY_CLIP}" \
            --curiosity "${curiosity}" \
            --curiosity_space "${CURIOSITY_SPACE}" \
            --curiosity_reference "${CURIOSITY_REFERENCE}" \
            --curiosity_schedule "${CURIOSITY_SCHEDULE}" \
            --curiosity_min "${CURIOSITY_MIN}" \
            --curiosity_cycles "${CURIOSITY_CYCLES}" \
            --plummer_power "${PLUMMER_POWER}" \
            --plummer_eps "${PLUMMER_EPS}" \
            --plummer_normalize "${PLUMMER_NORMALIZE}" \
            --plummer_terms "${PLUMMER_TERMS}" \
            ${fixed_latent_args[@]+"${fixed_latent_args[@]}"} \
            --history_interval "${HISTORY_INTERVAL}" \
            --output_dir "${out_dir}" \
            2>&1 | tee "${log_file}"
          echo "[$(date)] finished ${run_name}"
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
