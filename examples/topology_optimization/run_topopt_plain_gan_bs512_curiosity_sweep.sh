#!/usr/bin/env bash
set -euo pipefail

ITER="${ITER:-500}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BUFFER_MULTIPLIER="${BUFFER_MULTIPLIER:-2}"
FEM_WORKERS="${FEM_WORKERS:-8}"
G_LR="${G_LR:-0.03}"
D_LR="${D_LR:-0.03}"
OUT_ROOT="${OUT_ROOT:-results}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/fontcache}"

COMMON_ARGS=(
  --grid_width 40
  --grid_height 20
  --n_iter "${ITER}"
  --batch_size "${BATCH_SIZE}"
  --buffer_multiplier "${BUFFER_MULTIPLIER}"
  --latent_dim 64
  --encoding sorted_material
  --sorted_material_profile binary
  --density_filter_radius 0
  --projection_beta 0
  --initial_buffer_mode generator
  --generator_type conv
  --generator_output_norm centered_l2
  --generator_channels 64
  --discriminator_type mlp
  --discriminator_hidden_dims 128 128
  --optimizer_type default
  --discriminator_steps 1
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --g_lr "${G_LR}"
  --d_lr "${D_LR}"
  --fem_workers "${FEM_WORKERS}"
  --history_interval 25
  --seed "${SEED}"
)

run_variant() {
  local label="$1"
  shift
  local out_dir="${OUT_ROOT}/topopt_sorted_binary_default_gan_curiosity_${label}_bs${BATCH_SIZE}_buf$((BATCH_SIZE * BUFFER_MULTIPLIER))_glr${G_LR}_dlr${D_LR}_iter${ITER}_seed${SEED}"
  if [ "${SKIP_EXISTING}" = "true" ] && compgen -G "${out_dir}/top_designs_curiosity_*_seed_${SEED}.npz" > /dev/null; then
    echo "[$(date)] skipping existing ${label}: ${out_dir}"
    return
  fi
  echo "[$(date)] starting ${label}"
  uv run python examples/topology_optimization/cantilever_fem.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --output_dir "${out_dir}"
  echo "[$(date)] finished ${label}"
}

run_variant "raw0p001_buffer" \
  --curiosity 0.001 \
  --curiosity_space raw \
  --curiosity_reference buffer

run_variant "raw0p003_buffer" \
  --curiosity 0.003 \
  --curiosity_space raw \
  --curiosity_reference buffer

run_variant "plummer0p001_batchbuffer" \
  --curiosity 0.001 \
  --curiosity_space plummer \
  --curiosity_reference buffer \
  --plummer_terms batch_buffer \
  --plummer_power 1.0 \
  --plummer_eps 0.001 \
  --plummer_normalize layernorm

run_variant "plummer0p003_batchbuffer" \
  --curiosity 0.003 \
  --curiosity_space plummer \
  --curiosity_reference buffer \
  --plummer_terms batch_buffer \
  --plummer_power 1.0 \
  --plummer_eps 0.001 \
  --plummer_normalize layernorm

run_variant "chamferdiv0p03_ref32" \
  --curiosity 0 \
  --curiosity_space raw \
  --curiosity_reference buffer \
  --ladder_sequence diversity:0.03 \
  --diversity_reference_size 32 \
  --diversity_chamfer_max_points 256

run_variant "chamferdiv0p05_ref32" \
  --curiosity 0 \
  --curiosity_space raw \
  --curiosity_reference buffer \
  --ladder_sequence diversity:0.05 \
  --diversity_reference_size 32 \
  --diversity_chamfer_max_points 256
