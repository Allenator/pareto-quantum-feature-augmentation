# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quantum Feature Augmentation for Financial Market Prediction — YQuantum 2026 AWS x State Street Challenge. Investigates whether quantum-derived feature transformations improve out-of-sample financial prediction vs. classical baselines.

## Build & Run

Uses `uv` for package management with a local `.venv`.

```bash
uv sync                                              # install dependencies
uv run python scripts/run_synthetic.py static         # full static augmenter sweep
uv run python scripts/run_synthetic.py classical      # classical baselines only
uv run python scripts/run_synthetic.py trainable      # trainable augmenter sweep
uv run python scripts/plot_synthetic.py               # generate interactive plots
uv run python scripts/run_real.py quick               # real data: 3 tickers, monthly, minimal augmenters
uv run python scripts/run_real.py monthly             # real data: 10 tickers, monthly, all augmenters
uv run python scripts/run_real.py full                # real data: 10 tickers, daily, all augmenters
uv run python scripts/run_real.py ablation            # 2×2 factorial: regime features × corr quantum
```

## Architecture

- **`src/synthetic/`** — Modular experiment framework for the synthetic regime-switching task:
  - `config.py` — frozen dataclasses for all configuration (DGP, augmenters, models, run_id, features_dir)
  - `dgp.py` — data generation + parquet save/load
  - `augmenters/` — pluggable feature augmenters (classical, quantum fixed, quantum learned, neural). Each reports `n_trainable_params` and `n_random_params`.
  - `models/` — regression model wrappers (OLS, RidgeCV, LassoCV, ElasticNetCV). Report `coef_l2_norm` and `lasso_active_fraction`.
  - `evaluation/` — metrics (`metrics.py`), complexity metrics (`complexity.py`: effective rank, nonlinearity score, feature-target alignment), fairness checking and result aggregation (`comparison.py`)
  - `runner.py` — `ExperimentRunner` orchestrator with multiprocessing, tqdm progress, skip-if-exists caching, and feature matrix saving
- **`src/real/`** — Real S&P 500 excess return prediction pipeline:
  - `config.py` — `RealDataConfig` (tickers, DataBento dataset), `BacktestConfig` (window sizes in trading days), `ExperimentConfig`
  - `data.py` — DataBento OHLCV download, split adjustment, 14 stock-minus-market features, caching
  - `backtest.py` — `BacktestRunner` with walk-forward backtesting, monthly scaler caching, parallel quantum augmentation
  - `quantum_unified_real.py` — `UnifiedReservoirAugmenter`: generalized quantum reservoir for variable input dimension with modular/PCA/direct feature-to-qubit mapping
- **`scripts/`** — Entry points. `run_synthetic.py` and `run_real.py` configure and run experiments. `plot_synthetic.py` and `plot_real.py` generate interactive plots.
- **`data/synthetic/`** — Generated parquet datasets, tracked via **git LFS**. Do not regenerate unless the DGP changes.
- **`data/real/`** — Cached DataBento OHLCV parquet files, tracked via **git LFS**. Regenerated on first run or config change.
- **`features/synthetic/`** — Saved augmented feature matrices (NPZ), tracked via **git LFS**. Regenerated with results.
- **`results/synthetic/`** — Per-seed JSON results + `summary.csv`, scoped by `run_id`.
- **`results/real/`** — Per-(augmenter, model) JSON results, prediction parquets (LFS), and `summary.csv`, scoped by `run_id`.
- **`plots/synthetic/`** — Interactive HTML plots (Plotly).
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

Managed via `uv` and `pyproject.toml` with a local `.venv`. Key packages: `pennylane`, `amazon-braket-sdk`, `amazon-braket-pennylane-plugin`, `databento`, `scikit-learn`, `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `pyarrow`, `plotly`, `tqdm`.

DataBento API key stored in `.env.databento` (gitignored). Real data pipeline reads it automatically.
