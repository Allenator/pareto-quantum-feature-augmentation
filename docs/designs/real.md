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

Top 10 S&P 500 constituents by market cap: AAPL, MSFT, NVDA, AMZN, GOOGL, META, BRK.B, LLY, JPM, AVGO.

Data range: 2022-01-01 to 2025-12-31 (DataBento `XNAS.ITCH` ohlcv-1d). Market proxy: SPY (ETF). Stock splits are detected from price discontinuities and adjusted automatically. With a 2-year training warmup (504 trading days), out-of-sample evaluation starts ~mid-2024.

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

Training uses **rolling windows** in trading-day units, not random splits, to prevent look-ahead bias.

```
Timeline:  2022 ------------ mid-2024 ------------ 2025-12-31
           |<-- 2yr warmup -->|<---- OOS eval ---->|
           (504 trading days   (roll forward
            + feature lookback) 1 trading day)
```

At each evaluation date $t$:

1. **Train window**: all `(date, ticker)` rows in the preceding 504 trading days, ending 5 trading days before $t$
2. **Gap**: 5 trading days (= prediction horizon) between train end and test date
3. **Test**: all tickers on date $t$
4. **Model refit**: Ridge/Lasso refit from scratch each window

Training set size: 504 trading days $\times$ 10 tickers = ~5040 samples $\times$ 14 features per window.

### Cached Augmentation with Monthly Scaler Refit

Naive per-window augmentation is dominated by quantum circuit simulation: each window re-evaluates all circuits on ~5040 samples that are >99% identical to the previous window (503/504 days overlap). Two optimizations eliminate this redundancy:

**1. Persistent augmenter objects.** All augmenters are constructed once at the start and reused. This avoids rebuilding PennyLane QNodes, devices, and random weight arrays on every window. Quantum augmenters run in parallel via `ProcessPoolExecutor` (each worker rebuilds its own augmenter to avoid pickling QNode objects).

**2. Monthly scaler refit with full-dataset caching.** The `StandardScaler` is refit every 21 trading days (~monthly) instead of every window. On each refit, every augmenter transforms the **full dataset** once; results are cached as numpy arrays. Between refits, per-window work reduces to array slicing + Ridge/Lasso fitting (~ms).

**Compromise.** The scaler for windows between refits is "stale" — fit on one training window but applied to windows that have shifted by up to 20 trading days. Since the training window is 504 days and shifts by 1 day per step, the scaler mean/std changes by $\lesssim 0.4\%$ between refits. Empirically, this has negligible impact on results:

| Metric | Per-window scaler (baseline) | Monthly scaler (cached) |
|--------|------------------------------|-------------------------|
| MSE | 0.001616 | 0.001623 |
| IC | 0.412 | 0.407 |

**Performance.** On a 16-core machine with 10 quantum augmenters (8-qubit circuits), the cached approach is **~16x faster** than per-window augmentation. The precompute step (scaler refit + full-dataset augmentation of 8676 rows × 18 augmenters) takes ~20s, after which all 18 eval windows complete in <6s total.

### Time-Series Safeguards

| Concern | Mechanism |
|---------|-----------|
| No look-ahead in features | All features use only past data (lookback windows ending at or before $t$) |
| No look-ahead in target | Target is future 5-day return; never in training features |
| Train/test gap | 5 trading-day gap prevents leakage from prediction horizon |
| Scaler fit on train only | StandardScaler fit on training window, applied to full dataset for caching |
| PCA fit on train only | PCA mapping (when used) fit on training window at each scaler refit |
| Model refit each window | Ridge/Lasso refit from scratch every window |

### Aggregation

All per-window out-of-sample predictions are collected into a single series, then metrics (MSE, MAE, IC) are computed once on the full OOS series.

## Architecture

```
src/real/
  config.py                  # RealDataConfig, BacktestConfig, ExperimentConfig
  data.py                    # DataBento download + cache, feature engineering, build_dataset
  backtest.py                # BacktestRunner: walk-forward loop + metrics aggregation
  quantum_unified_real.py    # UnifiedReservoirAugmenter (generalized for variable n_features)
scripts/
  run_real.py                # Entry point (quick / full modes)
  plot_real.py               # OOS correlation, rolling IC, MSE bar chart
data/real/                   # Cached price parquet (git LFS)
results/real/                # JSON results + summary.csv
features/real/               # Saved augmented feature matrices
```

### Reused from `src/synthetic/`

| Component | Source |
|-----------|--------|
| `AugmenterConfig`, `ModelConfig` | `src/synthetic/config.py` |
| Classical augmenters (identity, polynomial, RFF, interaction_log) | `src/synthetic/augmenters/*` |
| Linear models (Ridge, Lasso) | `src/synthetic/models/*` |
| `_get_pairs()`, `_build_measurements()`, `_n_features()` | `src/synthetic/augmenters/quantum_unified.py` |
| `AugmenterResult`, `_make_result()` | `src/synthetic/augmenters/base.py` |
| `_build_augmenter()`, `_build_model()` | `src/synthetic/runner.py` |

## Dimensionality Strategy

With 14 input features (vs. 4 in synthetic), the winning quantum circuit design (4 qubits, hardcoded) cannot be used directly. `UnifiedReservoirAugmenter` in `src/real/quantum_unified_real.py` generalizes it with configurable feature-to-qubit mapping:

| Mapping | Description | Fit required |
|---------|-------------|-------------|
| `direct` | 1:1, `n_qubits = n_features` | No |
| `modular` | Average features into `n_qubits` bins: $x_j = \text{mean}(x_i : i \bmod n_q = j)$ | No |
| `pca` | PCA reduction to `n_qubits` dimensions | Yes (per scaler refit) |

Sweep `n_qubits` in $\{6, 8\}$ with observables in $\{Z, XYZ, Z{+}ZZ\}$ to study the tradeoff between circuit expressivity and feature count. Practical limit ~150 total features for ~5000 training samples.

## Augmenter Selection

### Classical

| Name | Method |
|------|--------|
| `identity` | Raw 14 features |
| `poly_deg2_interact` | Interaction terms only |
| `poly_deg2` | Full degree-2 polynomial |
| `interaction_log` | Interactions + log/abs |
| `rff_10` / `rff_30` | Random Fourier features |

### Quantum (Unified Reservoir)

All configs use the winning synthetic design: angle (RY) encoding, circular CNOT connectivity, random rotations. Modular feature-to-qubit mapping averages 14 input features into `n_qubits` bins. Implemented in `src/real/quantum_unified_real.py`.

| Name | Qubits | Observables | Total Features |
|------|--------|-------------|----------------|
| `qunified_z_6q_3L_3ens` | 6 | Z | 32 |
| `qunified_z_8q_3L_3ens` | 8 | Z | 38 |
| `qunified_z_10q_3L_3ens` | 10 | Z | 44 |
| `qunified_xyz_6q_3L_3ens` | 6 | XYZ | 68 |
| `qunified_xyz_8q_3L_3ens` | 8 | XYZ | 86 |
| `qunified_zzz_6q_3L_3ens` | 6 | Z+ZZ | 77 |
| `qunified_zzz_8q_3L_3ens` | 8 | Z+ZZ | 122 |
| `qunified_z_8q_3L_3ens_pca` | 8 | Z | 38 (PCA mapping) |

Ablations: depth (2L, 5L), connectivity (linear).

## Extension: Cross-Asset Correlation Features

The current pipeline treats each `(date, ticker)` row independently. Cross-asset correlation structure — how stocks co-move — is ignored. Three approaches for incorporating it, in order of complexity:

### Approach 1: Cross-Sectional Regime Features

Compute aggregate statistics across the 10-stock universe on each date and append as features shared by all tickers. These act as regime indicators:

| Feature | Formula | Signal |
|---------|---------|--------|
| `avg_corr_60d` | Mean of pairwise 60-day rolling return correlations across the 10 stocks | High = risk-off (stocks move together, excess returns are noise). Low = dispersion (stock-picking has signal) |
| `ret_dispersion` | Cross-sectional std of 5-day returns across stocks on each date | High dispersion = cross-sectional spread to exploit |
| `pca_ev1_share` | First eigenvalue of rolling 60-day return correlation matrix / trace | Measures single-factor dominance (market factor) |

These are scalars per date (same for all tickers), adding only 3 features. They give the model context: "is today an environment where stock-specific features are informative?"

Implementation: computed in `build_dataset()` after stacking all tickers, using cross-sectional return data. Appended as constant columns within each date.

### Approach 2: Relative Positioning Features

For each stock on each date, compute its rank or z-score within the cross-section of all 10 stocks:

| Feature | Formula | Signal |
|---------|---------|--------|
| Cross-sectional rank of `ret_5d` | $\text{rank}(ret_{5d,i,t})$ among all tickers on date $t$, scaled to $[0, 1]$ | Relative momentum position |
| Distance from cross-sectional mean | $ret_{5d,i,t} - \bar{ret}_{5d,t}$ | How far the stock is from the group |
| Lead-lag score | Rolling correlation of stock $i$'s returns with equal-weighted portfolio lagged returns | Whether stock leads or follows the group |

These add a "relative to peers" dimension beyond the single stock-minus-market reference point. Could be applied to any subset of the 14 existing features. Adds $k$ features per selected base feature.

Implementation: computed in `build_dataset()` after stacking, using `groupby("date")` to rank/z-score within each cross-section.

### Approach 3: Quantum Encoding of Correlation Structure

Instead of encoding a single stock's features into the quantum circuit, encode the **cross-sectional correlation structure** itself:

1. On each date $t$, compute the $N \times N$ rolling return correlation matrix $C_t$ across the $N = 10$ stocks (60-day window)
2. Extract the upper triangle (45 values), or the top-$k$ eigenvalues/eigenvectors
3. Feed this as input to a dedicated `UnifiedReservoirAugmenter` instance — the entanglement structure naturally maps to pairwise correlations
4. The circuit output becomes a set of "market structure features" appended to every stock's feature vector on that date

This is the most natural fit for quantum circuits: a $k$-qubit circuit represents $2^k$ amplitudes, which can encode exponentially many relationships between $k$ assets. The quantum circuit operates on the domain where it has the strongest theoretical advantage — extracting nonlinear structure from a correlation/covariance matrix.

$$
\phi_{q}^{corr}(C_t) = \langle \psi(\text{vech}(C_t)) | \hat{O} | \psi(\text{vech}(C_t)) \rangle
$$

where $\text{vech}(C_t)$ is the half-vectorization of the correlation matrix.

Implementation: a second augmenter instance in the backtest loop that takes correlation matrix features as input (shared across all stocks on the same date), with output appended alongside per-stock quantum features. The correlation matrix is computed from the training window only (no look-ahead).

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
uv run python scripts/run_real.py quick    # 3 tickers, monthly eval, identity + poly + 1 unified
uv run python scripts/run_real.py monthly  # 10 tickers, monthly eval, all 18 augmenters
uv run python scripts/run_real.py full     # 10 tickers, daily eval, all 18 augmenters
```
