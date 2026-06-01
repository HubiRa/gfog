# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `coco_bbob` example within the larger `gfog` (Gradient free optimization via gradients) workspace. The repository uses a workspace structure with the main `gfog` package and example projects like this one.

## Development Commands

**Environment Setup:**
- Create virtual environment: `uv venv && source .venv/bin/activate`
- Development install (from workspace root): `bash install.sh dev`
- Production install (from workspace root): `bash install.sh`

**Testing:**
- Run tests: `pytest -q` (from workspace root)
- Run this example: `python main.py`

**Dependencies:**
- Sync dependencies: `uv sync`
- This project depends on the `gfog` workspace package

## Architecture

- **Workspace Structure**: This is part of a `uv` workspace managed from `/Users/hubi/Work/gfog`
- **Main Package**: The `gfog` package is located at `src/gfog/` with subpackages for `opt/`, `curiosity/`, `buffer/`, `models/`
- **Rust Components**: The project includes Rust components via maturin/pyo3 in `src/gfog/buffer/buffer_core`
- **Build System**: Uses `uv` for Python dependency management and `maturin` for Rust-Python bindings

## Key Guidelines from AGENTS.md

- Use absolute imports (e.g., `from gfog.opt import ...`)
- Python: PEP 8, 4-space indents, type hints required for new code
- Functions/variables: `snake_case`, classes: `PascalCase`
- For Rust changes, rebuild with `bash install.sh dev` and verify with `bash install.sh`
- Run from workspace root for build/test commands