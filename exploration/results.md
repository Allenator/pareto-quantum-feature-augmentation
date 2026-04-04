# Exploration Results

Initial exploration run on 2026-04-04. All scripts executed with default parameters and random seed 42.

---

## e01 — Synthetic Data Generation

Generates 20,000 samples from the regime-switching DGP, split 10K train / 10K test (no shuffle).

| Metric | Train | Test |
|--------|-------|------|
| Shape | (10000, 4) | (10000, 4) |
| Regime 1 fraction | 75.58% | 74.75% |
| $Y$ mean | 2.6628 | 2.6671 |
| $Y$ std | 5.9450 | 5.9669 |

Regime proportions are close to the specified $P(\text{Regime 1}) = 0.75$. Train and test distributions are well-matched, confirming the DGP is stationary within each regime.

---

## e02 — Classical Baselines

Full 10K/10K train/test split. Features standardized via `StandardScaler` before fitting.

| Model | MSE | MAE | Corr |
|-------|-----|-----|------|
| LR (raw features) | 4.7692 | 1.6556 | 0.9307 |
| LR (poly deg=2) | 4.6475 | 1.1626 | 0.9357 |
| Ridge (poly deg=2) | 4.6445 | 1.1627 | 0.9358 |
| Lasso (poly deg=2, $\alpha=0.01$) | **4.3730** | **1.1691** | **0.9390** |

**Observations:**

- Polynomial expansion (degree 2) substantially reduces MAE (1.66 $\to$ 1.16) relative to raw features, confirming the DGP's nonlinear interactions ($X_1 \cdot X_3$ in Regime 2) are partially captured.
- Lasso outperforms Ridge, suggesting some polynomial terms are irrelevant (e.g., those involving $X_4$, the noise variable) and benefit from being zeroed out.
- Correlation is high across the board ($> 0.93$) because the linear component ($2X_1 - X_2$) dominates in 75% of samples.
- The residual MSE $\approx 4.4$ is close to the irreducible noise floor ($\text{Var}(\varepsilon) = 1$ per sample, but Regime 2's Cauchy-distributed $X_2$ and interaction terms inflate error).

---

## e03 — Quantum Feature Maps

Tested on 5 samples from a 100-sample dataset. All circuits use 4 qubits (one per feature) on `default.qubit` (statevector simulator, no shot noise).

### Angle Encoding + BasicEntanglerLayers

- **Output shape**: (5, 10) — 4 single-qubit $\langle Z_i \rangle$ + 6 pairwise $\langle Z_i Z_j \rangle$
- **Sample output**: values in $[-1, 1]$ as expected for Pauli expectation values
- **Weights**: 2 layers of random parameters (untrained)

### ZZ Feature Map

- **Output shape**: (5, 4) — single-qubit $\langle Z_i \rangle$ only
- **Config**: 2 repetitions of Hadamard + $R_Z$ encoding + CNOT-$R_Z$-CNOT entangling

### IQP Encoding

- **Output shape**: (5, 4) — single-qubit $\langle Z_i \rangle$ only
- **Sample output**: **identical to ZZ feature map**

**Bug: ZZ and IQP produce identical outputs.** Both circuits produce the same expectation values for every sample. This likely means the `IQPEmbedding` template with default parameters reduces to the same effective unitary as the hand-coded ZZ map at `reps=2`. The IQP circuit needs differentiated parameters (e.g., different `n_repeats`, edge patterns, or measurement observables) to produce distinct features.

---

## e04 — Hybrid Workflows

### VQC Feature Transformer

- **Training set**: 20 samples, 30 epochs, learning rate 0.05, 2 variational layers
- **Output**: 4 features (single-qubit $\langle Z_i \rangle$)

| Epoch | Cost (MSE proxy) |
|-------|-----------------|
| 10 | 32.43 |
| 20 | 30.85 |
| 30 | 30.26 |

**Observations:**

- Cost decreases but remains high — the circuit is far from fitting the data.
- Training on only 20 samples with gradient descent is insufficient. Needs larger batches and more epochs.
- The proxy loss (MSE on first qubit expectation) is a weak signal for learning useful features.

### Quantum Reservoir Computing

- **Config**: 3 random reservoirs, 3 layers each, different seeds
- **Output shape**: (5, 12) — 4 qubits $\times$ 3 reservoirs
- No training required (fixed random circuits).
- Reservoir features are diverse across reservoirs (different random unitaries), which is the intended behavior.

---

## e05 — Comparative Evaluation

Reduced dataset: 2,000 samples (1K train / 1K test) to keep quantum circuit execution tractable. Angle encoding uses random (untrained) weights with 2 layers.

| Model | MSE | MAE | Corr |
|-------|-----|-----|------|
| LR (raw) | 4.7979 | 1.6640 | 0.9342 |
| Ridge (poly deg=2) | **2.0084** | **1.0618** | **0.9730** |
| Ridge + Angle Encoding | 2.8427 | 1.2853 | 0.9616 |
| Ridge + ZZ Map | 3.9969 | 1.4859 | 0.9456 |

![Evaluation comparison bar chart](evaluation_comparison.png)

**Observations:**

- **Classical polynomial features dominate.** Ridge with degree-2 polynomials achieves the lowest MSE (2.01) and highest correlation (0.973), beating both quantum approaches.
- **Angle encoding is the best quantum approach** (MSE 2.84, Corr 0.962) but still 41% worse than classical poly on MSE. It benefits from producing 10 features (including pairwise $\langle Z_i Z_j \rangle$) vs. ZZ's 4.
- **ZZ map adds little value** over raw features (MSE 4.00 vs. 4.80). With only 4 output features (single-qubit expectations), it doesn't capture enough interaction structure.
- The smaller dataset (2K vs. 20K) inflates all MSE values compared to e02, but relative rankings are meaningful.

**Why quantum features underperform (expected at this stage):**

1. **Random weights** — angle encoding uses untrained parameters; the circuit isn't optimized for this data.
2. **Too few quantum features** — ZZ map produces only 4 features vs. polynomial expansion's 14 (for degree 2 with 4 inputs).
3. **No pairwise observables** in ZZ/IQP — only $\langle Z_i \rangle$ is measured, discarding entanglement information.
4. **Small sample size** — 1K training samples may not reveal quantum advantages.

---

## e06 — Real Stock Data

Downloads 3 tickers (AAPL, MSFT, NVDA) + S&P 500 index from 2020-01-01 to 2025-01-01 via yfinance. Feature construction demonstrated for AAPL.

| Metric | Value |
|--------|-------|
| Feature matrix shape | (1002, 11) |
| Date range | 2020-12-30 to 2024-12-23 |
| Target ($Y$) mean | 0.001512 |
| Target ($Y$) std | 0.025105 |

**Features constructed (11 total):**

| Feature | Type |
|---------|------|
| `ret_5d`, `ret_20d`, `ret_120d` | Price momentum |
| `max_high_20d`, `min_low_20d` | Price range |
| `rsi_10d` | Relative strength |
| `price_trend` | Moving average trend |
| `vol_z_5d`, `vol_z_20d` | Volume Z-scores |
| `vol_trend` | Volume moving average trend |
| `vol_trend_minus_price_trend` | Cross feature |

**Observations:**

- Data pipeline works end-to-end. ~1,000 valid trading days after dropping NaN warmup rows.
- Target is near-zero mean with low std (2.5%), consistent with excess returns being a mean-zero, low-SNR signal.
- Walk-forward backtesting logic is defined but not yet executed — the script only builds features and prints summary stats.
- No quantum features or model fitting are applied to real data yet.

---

## Summary of Issues and Next Steps

| Issue | Severity | Script |
|-------|----------|--------|
| ZZ and IQP circuits produce identical output | Bug | e03 |
| VQC training underfits (20 samples, 30 epochs) | Config | e04 |
| Quantum features use random/untrained weights in evaluation | Limitation | e05 |
| ZZ/IQP only measure single-qubit $\langle Z_i \rangle$ (no pairwise) | Design gap | e03, e05 |
| Walk-forward backtest not yet executed | Incomplete | e06 |
| Quantum features not integrated into real data pipeline | Incomplete | e06 |
