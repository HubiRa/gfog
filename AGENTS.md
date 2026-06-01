# AGENTS.md

## Build & Test
- Dev install: `bash install.sh dev` (requires Python 3.13+, Rust, `uv`, `maturin`)
- Run all tests: `pytest -q`
- Run single test: `pytest tests/test_foo.py::test_bar -v`
- Run example: `python examples/testfunctions/example_himmelblau.py`

## Lint & Format
- Ruff via pre-commit: `ruff check --fix` and `ruff format`
- Rust: `cargo fmt` and `cargo clippy` in `src/gfog/buffer/buffer_core/`

## Code Style
- Type hints required; use `dataclass` for config/component structs
- Naming: `snake_case` (files/functions/vars), `PascalCase` (classes)
- Imports: absolute paths (`from gfog.opt import ...`), relative within package (`from ..buffer import Buffer`)
- Errors: raise `ValueError` with descriptive messages for invalid configs
- Docstrings: concise Google-style, one-liner for simple classes

## Project Structure
- Source: `src/gfog/` — `opt/`, `curiosity/`, `buffer/`, `models/`
- Rust core: `src/gfog/buffer/buffer_core/` (pyo3 bindings)
- Examples: `examples/`; tests: `tests/test_*.py`
