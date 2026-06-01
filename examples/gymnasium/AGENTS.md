# Repository Guidelines

Concise guide for contributors to build, test, and extend the project safely and consistently.

## Project Structure & Module Organization
- Source: `src/gfog/` with subpackages `opt/`, `curiosity/`, `buffer/`, `models/`.
- Rust core: `src/gfog/buffer/buffer_core/` (maturin + pyo3).
- Public API: `src/gfog/__init__.py`.
- Examples: `examples/`; assets: `assets/`; tests: `tests/`.
- Imports: prefer absolute (e.g., `from gfog.opt import ...`).

## Build, Test, and Development Commands
- Requirements: Python 3.13+, Rust toolchain, `uv`, `maturin`.
- Create venv: `uv venv && source .venv/bin/activate`.
- Dev install: `bash install.sh dev` (editable Python + Rust develop build).
- Build wheels: `bash install.sh` (release wheels for production).
- Run tests: `pytest -q` (quiet, fast feedback).
- Run example: `python examples/testfunctions/example_himmelblau.py`.
- Run CartPole (DefaultOpt): `python examples/gymnasium/cartpole.py`.
- Run CartPole (DeltaOpt): `python examples/gymnasium/cartpole_delta.py`.
- Run HalfCheetah Vector (DefaultOpt): `python examples/gymnasium/halfcheetah_vector.py`.
- Run HalfCheetah Vector (DeltaOpt): `python examples/gymnasium/halfcheetah_vector_delta.py`.
- Run Humanoid Vector (DefaultOpt): `python examples/gymnasium/humanoid_vector.py`.
- Run Humanoid Vector (DeltaOpt): `python examples/gymnasium/humanoid_vector_delta.py`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indents; add type hints for new/edited code.
- Names: files/modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.
- Docstrings: concise, Google-style; include brief module docstrings when adding modules.
- Rust: edition 2021; prefer small functions; format with `cargo fmt`.

## Testing Guidelines
- Framework: `pytest`; place tests under `tests/` as `test_*.py` with `test_*` functions.
- Scope: target core logic (buffer sorting, curiosity scheduling, optimizer steps).
- Determinism: avoid flakiness and heavy I/O; seed randomness when relevant.
- Quick run before PR: `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: short, present tense, focused (e.g., "add pix2pix to init").
- PRs: clear description, linked issues, repro steps; add screenshots/GIFs for example or asset changes.
- Note scope (Python/Rust) and any API/behavior changes; confirm tests pass locally.

## Security & Configuration Tips
- Never commit secrets. Pin runtime deps in `pyproject.toml`; sync with `uv sync`.
- For Rust changes, rebuild with `bash install.sh dev` and verify prod wheels with `bash install.sh` before merge.
