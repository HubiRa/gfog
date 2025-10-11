#!/usr/bin/env bash
set -euo pipefail

DEV_MODE=false
if [[ "${1:-}" == "dev" ]]; then
  DEV_MODE=true
fi

if [ "$DEV_MODE" = true ]; then
  echo "Installing in development mode..."
  uv sync --all-groups
  uv pip install -e .
  maturin develop -m src/gfog/buffer/buffer_core/Cargo.toml
  pre-commit install
else
  echo "Installing in production mode..."
  uv sync
  uv pip install .
  maturin build --release -m src/gfog/buffer/buffer_core/Cargo.toml
  uv pip install src/gfog/buffer/buffer_core/target/wheels/*.whl
fi
