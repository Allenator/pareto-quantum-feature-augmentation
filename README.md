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
exploration/          # Runnable exploration scripts (partitioned from reference notebook)
  e00_common.py       # Shared imports and utilities
  e01_data_generation.py    # Synthetic regime-switching data
  e02_classical_baseline.py # Polynomial features + linear regression
  e03_quantum_feature_maps.py # Angle, ZZ, IQP encoding circuits
  e04_hybrid_workflows.py   # VQC + quantum reservoir computing
  e05_evaluation.py         # Side-by-side comparison
  e06_real_stock_data.py    # S&P 500 feature construction + walk-forward
docs/                 # Project documentation (see docs/INDEX.md)
AWS-State-Street-Challenge/  # Challenge repo (git submodule)
```

## Running exploration scripts

```bash
# From project root
uv run python -m exploration.e01_data_generation
uv run python -m exploration.e02_classical_baseline
```

## Challenge Reference

See [docs/specs/challenge.md](docs/specs/challenge.md) for the full challenge specification, or the original in [AWS-State-Street-Challenge/README.md](AWS-State-Street-Challenge/README.md).
