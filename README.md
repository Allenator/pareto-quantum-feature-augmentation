# Quantum Feature Augmentation for Financial Market Prediction

YQuantum 2026 — AWS x State Street Challenge

## Overview

This project investigates whether quantum-derived feature transformations improve out-of-sample predictive performance for financial prediction tasks, compared to classical feature engineering under strict train/test separation and rolling backtests.

Two prediction tasks:
1. **Synthetic regime-switching process** — controlled, interpretable evaluation
2. **S&P 500 stock excess returns** — real-world financial data with walk-forward backtesting

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Project Structure

```
src/synthetic/             # Modular experiment framework
  config.py                # Experiment configuration (dataclasses)
  dgp.py                   # Data generation + parquet save/load
  augmenters/              # Pluggable feature augmenters
    classical.py           # Polynomial, log/abs, RFF, oracle
    quantum_fixed.py       # Angle encoding, ZZ, IQP, reservoir, QAOA
    quantum_learned.py     # VQC with trainable weights
    neural.py              # MLP, autoencoder, learned RFF (torch)
  models/                  # Regression model wrappers
    linear.py              # OLS, RidgeCV, LassoCV, ElasticNetCV
  evaluation/              # Metrics, complexity, fairness, result aggregation
    complexity.py          # Effective rank, nonlinearity, feature-target alignment
  runner.py                # ExperimentRunner orchestrator
scripts/
  run_synthetic.py         # Entry point for synthetic experiments
  plot_synthetic.py        # Interactive Plotly visualizations
data/synthetic/            # Generated datasets (parquet, git LFS)
features/synthetic/        # Saved feature matrices (NPZ, git LFS)
results/synthetic/         # Experiment results (JSON + summary CSV)
plots/synthetic/           # Interactive HTML plots
exploration/               # Legacy exploration scripts (from reference notebook)
docs/                      # Documentation (see docs/INDEX.md)
AWS-State-Street-Challenge/  # Challenge repo (git submodule)
```

## Running Experiments

```bash
# Static augmenter sweep — all fixed methods (main experiment)
uv run python scripts/run_synthetic.py static

# Trainable augmenter sweep — MLP, learned RFF, autoencoder
uv run python scripts/run_synthetic.py trainable

# Classical baselines only (fast)
uv run python scripts/run_synthetic.py classical

# Generate interactive plots
uv run python scripts/plot_synthetic.py
```

## Challenge Reference

See [docs/specs/challenge.md](docs/specs/challenge.md) for the full challenge specification, or the original in [AWS-State-Street-Challenge/README.md](AWS-State-Street-Challenge/README.md).
