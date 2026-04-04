# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quantum Feature Augmentation for Financial Market Prediction — YQuantum 2026 AWS x State Street Challenge. Investigates whether quantum-derived feature transformations improve out-of-sample financial prediction vs. classical baselines.

## Build & Run

Uses `uv` for package management with a local `.venv`.

```bash
uv sync                                              # install dependencies
uv run python scripts/run_synthetic.py classical      # classical baselines only
uv run python scripts/run_synthetic.py full           # full experiment (slow)
uv run python -m exploration.e01_data_generation      # legacy exploration scripts
```

## Architecture

- **`src/synthetic/`** — Modular experiment framework for the synthetic regime-switching task:
  - `config.py` — frozen dataclasses for all configuration
  - `dgp.py` — data generation + parquet save/load
  - `augmenters/` — pluggable feature augmenters (classical, quantum fixed, quantum learned, neural)
  - `models/` — regression model wrappers (OLS, RidgeCV, LassoCV, ElasticNetCV)
  - `evaluation/` — metrics, fairness checking, result aggregation
  - `runner.py` — `ExperimentRunner` orchestrator
- **`scripts/`** — Entry points. `run_synthetic.py` configures and runs experiments.
- **`data/synthetic/`** — Generated parquet datasets, tracked via **git LFS**. Do not regenerate unless the DGP changes.
- **`results/synthetic/`** — Per-seed JSON results + `summary.csv`. Regenerated on each run.
- **`exploration/`** — Legacy exploration scripts from the reference notebook. Numbered `e00`–`e06`.
- **`docs/`** — Project documentation. **All docs must be indexed in `docs/INDEX.md`**. Update the index each time there is a meaningful change in documentation.
- **`AWS-State-Street-Challenge/`** — Upstream challenge repo (git submodule). Read-only reference.

## Key Constraints (from challenge spec)

- Linear regression is the **mandatory** baseline model
- The **same model class** must be used for classical and quantum features (isolates feature effect)
- Regularization (Ridge/Lasso) must be applied **consistently** across both feature sets
- Real data task uses **walk-forward backtesting** (rolling 2-year window, no look-ahead)
- All features for real data are **stock minus market** values
- Quantum resource usage (circuit depth, qubit count, memory) is a **required result dimension**

## Conventions

- Write all math in documentation using LaTeX notation. Use `$...$` for inline math. Use the multiline form for display math:
  ```
  $$
  <math>
  $$
  ```

## Dependencies

Managed via `uv` and `pyproject.toml` with a local `.venv`. Key packages: `pennylane`, `amazon-braket-sdk`, `amazon-braket-pennylane-plugin`, `scikit-learn`, `torch` (MPS backend for M4 Max), `yfinance`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `pyarrow`.
