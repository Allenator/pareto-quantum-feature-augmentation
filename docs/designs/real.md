# Real Financial Data Experiment Design

## Context

This document describes the architecture for evaluating quantum feature augmentation on real S&P 500 stock excess returns, the second task in the YQuantum 2026 challenge. The framework reuses augmenters, models, and evaluation from `src/synthetic/`, adding a data pipeline, feature engineering, and walk-forward backtesting.

## Target Variable

5-day forward excess return of stock $i$ relative to the S&P 500 index:

$$
Y_{i,t} = R_{i,t \to t+5}^{close} - R_{SP500,t \to t+5}^{close}
$$

where $R_{i,t \to t+5}^{close} = P_{i,t+5}^{close} / P_{i,t}^{close} - 1$. This removes market-wide effects and focuses on relative outperformance.

## Stock Universe

Top 10 S&P 500 constituents by market cap: AAPL, MSFT, NVDA, AMZN, GOOGL, META, BRK-B, LLY, JPM, AVGO.

Data range: 2022-01-01 to 2025-12-31 (yfinance). With a 2-year training warmup, out-of-sample evaluation starts ~2024-01-01.

## Feature Construction

All features are **stock minus market** cross-sectional statistics, computed from rolling lookback windows at each date $t$. Each observation is a `(date, ticker)` pair with 14 features — structurally analogous to the synthetic task's 4 features per observation.

### Price Features

| Feature | Lookback | Formula |
|---------|----------|---------|
| `ret_5d` | 5d | $R_{stock}^{5d} - R_{market}^{5d}$ |
| `ret_20d` | 20d | $R_{stock}^{20d} - R_{market}^{20d}$ |
| `ret_120d` | 120d | $R_{stock}^{120d} - R_{market}^{120d}$ |
| `high_20d` | 20d | $(\max(H_{20d})/P - 1)_{stock} - (\ldots)_{market}$ |
| `low_20d` | 20d | $(\min(L_{20d})/P - 1)_{stock} - (\ldots)_{market}$ |
| `rsi_10d` | 10d | $RSI_{stock}^{10} - RSI_{market}^{10}$ |
| `price_trend` | 50d | $(MA_{10}/MA_{50} - 1)_{stock} - (\ldots)_{market}$ |

### Volume Features

| Feature | Lookback | Formula |
|---------|----------|---------|
| `vol_z_5d` | 5d+lookback | $\text{zscore}(\bar{V}_{5d})_{stock} - \text{zscore}(\bar{V}_{5d})_{market}$ |
| `vol_z_20d` | 20d+lookback | $\text{zscore}(\bar{V}_{20d})_{stock} - \text{zscore}(\bar{V}_{20d})_{market}$ |
| `vol_trend` | 50d | $(MA_{10}^{vol}/MA_{50}^{vol} - 1)_{stock} - (\ldots)_{market}$ |

### Price x Volume Combined Features

| Feature | Lookback | Formula |
|---------|----------|---------|
| `vwret_5d` | 5d | Volume-weighted return $- R_{5d}$ (stock $-$ market) |
| `ret_x_vol` | 5d | $ret_{5d} \times \text{zscore}(V_{5d})$ (stock $-$ market) |
| `rsi_x_vol` | 10d+5d | $RSI_{10} \times \text{zscore}(V_{5d})$ (stock $-$ market) |
| `vol_minus_price_trend` | 50d | $vol\_trend - price\_trend$ (stock $-$ market) |

Maximum lookback is 120 trading days (`ret_120d`). The volume z-score uses a 252-day rolling window for the mean/std normalization.

## Walk-Forward Backtesting

Training uses **rolling windows**, not random splits, to prevent look-ahead bias.

```
Timeline:  2022 ------------ 2024 ------------ 2025-12-31
           |<-- 2yr warmup -->|<-- OOS eval -->|
           (features need     (roll forward
            120d lookback)     1 day at a time)
```

At each evaluation date $t$:

1. **Train window**: all `(date, ticker)` rows in $[t - 2\text{yr},\; t - 5\text{d})$
2. **Gap**: 5 trading days (= prediction horizon) between train end and test date
3. **Test**: all tickers on date $t$
4. **Refit**: StandardScaler, augmenters, and models are fit from scratch each window

Training set size: ~500 trading days $\times$ 10 tickers = ~5000 samples $\times$ 14 features per window.

### Time-Series Safeguards

| Concern | Mechanism |
|---------|-----------|
| No look-ahead in features | All features use only past data (lookback windows ending at or before $t$) |
| No look-ahead in target | Target is future 5-day return; never in training features |
| Train/test gap | 5-day gap prevents leakage from prediction horizon |
| Refit each window | Scaler, augmenters, models refit from scratch (no carry-over) |

### Aggregation

All per-window out-of-sample predictions are collected into a single series, then metrics (MSE, MAE, IC) are computed once on the full OOS series.

## Architecture

```
src/real/
  config.py           # RealDataConfig, BacktestConfig, ExperimentConfig
  data.py             # yfinance download + cache, feature engineering, build_dataset
  backtest.py         # BacktestRunner: walk-forward loop + metrics aggregation
scripts/
  run_real.py          # Entry point (quick / full modes)
  plot_real.py         # OOS correlation, rolling IC, MSE bar chart
data/real/             # Cached price parquet
results/real/          # JSON results + summary.csv
features/real/         # Saved augmented feature matrices
```

### Reused from `src/synthetic/`

| Component | Source |
|-----------|--------|
| `AugmenterConfig`, `ModelConfig` | `src/synthetic/config.py` |
| All augmenters (classical, quantum, neural) | `src/synthetic/augmenters/*` |
| All linear models (OLS, Ridge, Lasso, ElasticNet) | `src/synthetic/models/*` |
| `compute_metrics()`, `ExperimentMetrics` | `src/synthetic/evaluation/metrics.py` |
| `ResultTable` | `src/synthetic/evaluation/comparison.py` |
| `_build_augmenter()`, `_build_model()` | `src/synthetic/runner.py` |

## Dimensionality Strategy

With 14 input features (vs. 4 in synthetic), quantum circuits need more qubits or feature-to-qubit mapping:

- All existing augmenters accept `n_qubits` independently of input dimension
- When `n_qubits < n_features`: features map to qubits via modular wrapping ($X_i \to$ qubit $i \bmod n_q$, angles summed)
- Sweep `n_qubits` in $\{4, 6, 8, 10, 14\}$ to study the tradeoff
- Reservoir augmenters with multiple reservoirs are particularly well-suited (independent random parameters)

## Augmenter Selection

### Classical

| Name | Method |
|------|--------|
| `identity` | Raw 14 features |
| `poly_deg2_interact` | Interaction terms only |
| `poly_deg2` | Full degree-2 polynomial |
| `interaction_log` | Interactions + log/abs |
| `rff_10` / `rff_30` | Random Fourier features |

### Quantum

| Name | Qubits | Observables | Notes |
|------|--------|-------------|-------|
| `reservoir_3x3_Z_4q` | 4 | Z | Baseline, 4-qubit mapping |
| `reservoir_3x3_XYZ_4q` | 4 | XYZ | Multi-basis, 4-qubit |
| `reservoir_3x3_XYZ_8q` | 8 | XYZ | Mid-range qubit count |
| `reservoir_3x3_XYZ_14q` | 14 | XYZ | 1:1 feature-qubit mapping |
| `angle_strong_8q_2L` | 8 | Z+ZZ | StronglyEntangling |
| `angle_strong_14q_2L` | 14 | Z+ZZ | Full feature encoding |

## Evaluation

Per the challenge spec (Sections 14-15):

- **Aggregate OOS metrics**: MSE, MAE, information coefficient (Pearson $\rho$)
- **Time stability**: rolling 3-month IC
- **Feature count vs performance**: scatter plot
- **Overfitting diagnostics**: train vs test MSE gap
- **Quantum resource usage**: circuit depth, qubit count, gate count
- **Cost-performance tradeoffs**: wall clock, quantum resources vs MSE

## Running

```bash
uv run python scripts/run_real.py quick    # 3 tickers, monthly eval, identity + poly + 1 reservoir
uv run python scripts/run_real.py full     # 10 tickers, daily eval, all augmenters
```
