#!/bin/bash

# Parse command line arguments
DEV_MODE=false
if [[ "$1" == "dev" ]]; then
    DEV_MODE=true
fi

# Install dependencies
uv sync

# Install Python package
if [ "$DEV_MODE" = true ]; then
    echo "Installing in development mode..."
    uv pip install -e .
    maturin develop -m src/gfog/buffer/buffer_core/Cargo.toml
else
    echo "Installing in production mode..."
    uv pip install .
    maturin build --release -m src/gfog/buffer/buffer_core/Cargo.toml
    uv pip install src/gfog/buffer/buffer_core/target/wheels/*.whl
fi
