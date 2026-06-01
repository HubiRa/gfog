#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-500}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-8}"
FEM_WORKERS="${FEM_WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-results}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

COMMON_ARGS=(
  --preset tom_cantilever_2d
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding sorted_material
  --sorted_material_profile binary
  --ranker_list_size 64
  --ranker_sample_pool_size 128
  --ranker_steps 1
  --discriminator_steps 1
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --g_lr 0.03
  --d_lr 0.03
  --generator_type conv
  --generator_channels 64
  --discriminator_type mlp
  --discriminator_hidden_dims 128 128
  --density_filter_radius 0
  --projection_beta 0
  --fem_workers "${FEM_WORKERS}"
  --history_interval 50
  --seed "${SEED}"
)

python examples/topology_optimization/cantilever_fem.py \
  "${COMMON_ARGS[@]}" \
  --optimizer_type quantile_ranked_default \
  --ranker_weight 1.0 \
  --ranker_target_curve exp \
  --ranker_tau 4 \
  --ranker_target_scope local \
  --curiosity 0 \
  --output_dir "${OUT_ROOT}/tom_cantilever_sorted_binary_quantile_tau4_iter${ITER}_seed${SEED}"

python examples/topology_optimization/cantilever_fem.py \
  "${COMMON_ARGS[@]}" \
  --optimizer_type hybrid_contextual_utility \
  --utility_target_scale 100 \
  --utility_loss smooth_l1 \
  --utility_weight 0.1 \
  --generator_utility_weight 0.1 \
  --utility_clip 3 \
  --curiosity 0.3 \
  --curiosity_space raw \
  --curiosity_reference batch \
  --curiosity_schedule warmup_cosine_annealing \
  --curiosity_warmup_frac 0 \
  --curiosity_min 0.0333333333 \
  --curiosity_cycles 4 \
  --d_score_center_weight 0.01 \
  --d_score_scale_weight 0.01 \
  --d_score_target_std 1 \
  --output_dir "${OUT_ROOT}/tom_cantilever_sorted_binary_hybrid_contextual_utility_iter${ITER}_seed${SEED}"
