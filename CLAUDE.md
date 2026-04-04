# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quantum Feature Augmentation for Financial Market Prediction — YQuantum 2026 AWS x State Street Challenge. Investigates whether quantum-derived feature transformations improve out-of-sample financial prediction vs. classical baselines.

## Build & Run

```bash
uv sync                                    # install dependencies
uv run python -m exploration.e01_data_generation   # run any script as module
uv run python -m exploration.e02_classical_baseline
```

All exploration scripts live in `exploration/` and are runnable via `python -m exploration.<module_name>`. Cross-script imports go through explicit module paths (e.g., `from exploration.e01_data_generation import ...`).

## Architecture

- **`exploration/`** — Runnable Python scripts partitioned from the reference notebook (`AWS-State-Street-Challenge/QFA_Overview.ipynb`). Numbered `e00`–`e06`, each covering one phase: data generation, classical baseline, quantum feature maps, hybrid workflows, evaluation, real stock data.
- **`docs/`** — Project documentation. **All docs must be indexed in `docs/INDEX.md`**. Update the index each time there is a meaningful change in documentation.
- **`docs/specs/`** — Challenge specifications and requirements.
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

Managed via `uv` and `pyproject.toml`. Key packages: `pennylane`, `amazon-braket-sdk`, `amazon-braket-pennylane-plugin`, `scikit-learn`, `yfinance`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.
