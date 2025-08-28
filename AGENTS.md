# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/lerobot/` (core packages, scripts under `lerobot.scripts`).
- Tests: `tests/` (mirrors package layout, e.g., `tests/motors/test_*.py`).
- Docs & examples: `docs/`, `examples/`; benchmarks in `benchmarks/`.
- Dev tooling: `pyproject.toml` (ruff, mypy, bandit), `.pre-commit-config.yaml`, `Makefile`.

## Build, Test, and Development Commands
- Setup (editable + dev tools): `pip install -e .[dev,test]`.
- Lint/format (auto-fix): `pre-commit run -a` (installs with `pre-commit install`).
- Type check: `mypy` (targets configured modules under `src/lerobot`).
- Unit tests: `pytest -q` or `pytest tests/<area>`; coverage: `pytest --cov=src/lerobot`.
- E2E smoke tests (lightweight configs): `make test-end-to-end` (see `Makefile`).
- CLI examples: `lerobot-train ...`, `lerobot-eval ...` (see `pyproject.toml` entry points).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, max line length 110 (ruff).
- Imports: isort via ruff; first‑party is `lerobot`.
- Strings: prefer double quotes (ruff formatter).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Docstrings: Google style (configured; gradually enforced).

## Testing Guidelines
- Framework: `pytest` with fixtures under `tests/fixtures` and `conftest.py`.
- Layout: put tests alongside domain folders (e.g., `tests/policies/test_*.py`).
- Naming: files `test_*.py`; tests should be deterministic and independent.
- Skips/marks: respect platform/hardware markers in `tests/utils.py`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; include scope tags when helpful (e.g., `[policies]`, `[envs]`).
- Link issues with `#<id>`; keep changes focused and well-scoped.
- Before pushing: run `pre-commit run -a` and `pytest` locally.
- PRs: clear description, rationale, minimal repro or usage snippet, screenshots/logs when UI/vis changes, note docs updates (`docs/`) and tests added.

## Security & CI Tips
- Secrets: scans via `gitleaks`; do not commit tokens or private data.
- Static checks: `bandit -c pyproject.toml`; fix or justify ignores.
- Type coverage is being expanded with `mypy`; keep new modules typed where practical.
