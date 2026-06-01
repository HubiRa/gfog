# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/gfog/` with subpackages: `opt/`, `curiosity/`, `buffer/`, `models/`.
- Rust buffer core (maturin + pyo3): `src/gfog/buffer/buffer_core`.
- Public API: `src/gfog/__init__.py`.
- Examples: `examples/`; assets: `assets/`; tests: `tests/`.
- Prefer absolute imports (e.g., `from gfog.opt import ...`).

## Build, Test, and Development Commands
- Environment: Python 3.13+, Rust toolchain, `uv`, `maturin`.
- Create venv: `uv venv && source .venv/bin/activate`.
- Dev install (editable + Rust develop): `bash install.sh dev`.
- Prod wheels/build: `bash install.sh`.
- Run tests: `pytest -q`.
- Run an example: `python examples/testfunctions/example_himmelblau.py`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indents; type hints required for new code.
- Names: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.
- Docstrings: concise, Google‑style; add brief module docstrings for new modules.
- Imports: use absolute package paths under `gfog`.
- Rust: edition 2021; keep functions small; run `cargo fmt` if available.

## Testing Guidelines
- Framework: `pytest` with deterministic tests; avoid heavy I/O.
- Location: under `tests/`; files `test_*.py`, tests `test_*`.
- Focus: core logic (buffer sorting, curiosity scheduling, optimizer steps).
- Run locally: `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: short, present tense, focused (e.g., "add pix2pix to init").
- PRs: clear description, linked issues, and repro steps; attach screenshots/GIFs for example/asset changes.
- Note scope (Python/Rust) and any API/behavior changes.

## Security & Configuration Tips
- Do not commit secrets or credentials.
- Pin runtime dependencies in `pyproject.toml`; sync with `uv sync`.
- For Rust changes, rebuild in dev (`bash install.sh dev`) and verify prod wheels (`bash install.sh`) before merging.
