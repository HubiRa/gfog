#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/overnight_logs

run() {
  local name="$1"
  shift
  echo "[$(date)] starting ${name}"
  "$@" 2>&1 | tee "results/overnight_logs/${name}.log"
  echo "[$(date)] finished ${name}"
}

BASE=(
  python examples/topology_optimization/cantilever_fem.py
  --grid_width 40
  --grid_height 20
  --encoding topk_volume
  --optimizer_type lsgan
  --curiosity 0
  --density_filter_radius 1
  --projection_beta 1
  --seed 0
  --g_torch_optimizer muon
  --d_torch_optimizer muon
  --g_lr 0.03
  --d_lr 0.03
)

run topk_muon_muon_iter10000_seed0 \
  "${BASE[@]}" \
  --n_iter 10000 \
  --batch_size 64 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter10000_gmuon0p03_dmuon0p03_seed0

run topk_muon_muon_iter3000_g256x3_seed0 \
  "${BASE[@]}" \
  --n_iter 3000 \
  --batch_size 64 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --generator_hidden_dims 256 256 256 \
  --discriminator_hidden_dims 128 128 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter3000_g256x3_gmuon0p03_dmuon0p03_seed0

run topk_muon_muon_iter3000_batch128_seed0 \
  "${BASE[@]}" \
  --n_iter 3000 \
  --batch_size 128 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter3000_batch128_gmuon0p03_dmuon0p03_seed0

run topk_muon_muon_iter3000_seed1 \
  "${BASE[@]}" \
  --n_iter 3000 \
  --batch_size 64 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --seed 1 \
  --output_dir results/fem_cantilever_topkvolume_lsgan_iter3000_gmuon0p03_dmuon0p03_seed1
