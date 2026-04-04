# Challenge Specification: Quantum Feature Augmentation for Financial Market Prediction

**Event**: YQuantum 2026 — AWS x State Street Challenge

## Core Scientific Question

> Do quantum-derived feature transformations improve out-of-sample predictive performance for financial prediction tasks, relative to classical feature engineering, when evaluated under strict train/test separation and rolling backtests?

Both positive and negative results are valid. What matters is **experimental rigor, transparency, and reproducibility**.

---

## Part I — Synthetic Regime-Switching Process

### Data Generating Process (DGP)

Target variable $Y$ is generated from a latent regime-switching model:

$$
Y = \begin{cases} 2X_{1} - X_{2} + \varepsilon, & \text{if Regime 1} \\ (X_{1} \cdot X_{3}) + \log(|X_{2}| + 1) + \varepsilon, & \text{if Regime 2} \end{cases}
$$

**Regime probabilities:**

$$
P(\text{Regime 1}) = 0.75, \quad P(\text{Regime 2}) = 0.25
$$

The regime indicator is **latent** (not provided by default).

### Feature Distributions

**Regime 1:**

$$
X_{1}, X_{2}, X_{3} \sim \mathcal{N}(0,1), \quad X_{4} \sim \text{Uniform}(-1,1), \quad \varepsilon \sim \mathcal{N}(0,1)
$$

**Regime 2:**

$$
(X_{1}, X_{3}) \sim \mathcal{N}\!\left(\begin{bmatrix} 3 \\ 3 \end{bmatrix}, \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix}\right), \quad X_{2} \sim \text{Cauchy}(0,1), \quad X_{4} \sim \text{Exp}(\lambda = 1), \quad \varepsilon \sim \mathcal{N}(0,1)
$$

| Variable | Purpose |
|----------|---------|
| $X_{1}$ | Linear signal in Regime 1 |
| $X_{2}$ | Linear (R1), nonlinear heavy-tailed (R2) |
| $X_{3}$ | Only relevant in Regime 2 |
| $X_{4}$ | Pure noise |
| Regime | Hidden non-stationarity |

### Experimental Setup

- Training set: 10,000 observations
- Test set: 10,000 observations
- Optional validation set for tuning
- Multiple random seeds encouraged

### Metrics

- **MSE**: $\text{MSE} = \frac{1}{N} \sum_{i} (Y_{i} - \hat{Y}_{i})^{2}$
- **MAE**: $\text{MAE} = \frac{1}{N} \sum_{i} |Y_{i} - \hat{Y}_{i}|$
- **Correlation**: $\rho(Y, \hat{Y})$

---

## Part II — Predicting Stock Excess Returns

### Target Variable

Next 5-day excess returns of S&P 500 constituents:

$$
Y_{i,t} = R_{i,t}^{(5d)} - R_{\text{SP500},t}^{(5d)}
$$

Start with the ~10 largest stocks; expand if time permits.

### Feature Construction

All features defined as **stock minus market** values, using yfinance data (price + volume).

**Price-related examples**:
- Prior 5/20/120-day return (stock $-$ market)
- $\frac{\max(\text{High}_{20d})}{\text{Price}} - 1$ (stock $-$ market)
- $\frac{\min(\text{Low}_{20d})}{\text{Price}} - 1$ (stock $-$ market)
- RSI over 10 days (stock $-$ market)
- Price trend: $\frac{MA_{10}}{MA_{50}} - 1$ (stock $-$ market)

**Volume-related examples**:
- Z-score of 5/20-day average volume (stock $-$ market)
- Volume trend: $\frac{MA_{10}}{MA_{50}} - 1$ (stock $-$ market)

**Price $\times$ Volume examples**:
- Volume-weighted return: $\sum_{d=1}^{5} w_{d} \cdot r_{d}$, where $w_{d} = \frac{V_{d}}{\sum_{k=1}^{5} V_{k}}$ (stock $-$ market)
- Volume-weighted return $-$ actual return (stock $-$ market)
- Prior 5-day return $\times$ Z-score of 5-day volume (stock $-$ market)
- RSI $\times$ Z-score of 5-day volume (stock $-$ market)
- Volume trend $-$ price trend (stock $-$ market)

### Walk-Forward Backtest

At each time $t$:
1. Train on past 2 years
2. Validate (optional)
3. Predict next 5-day return
4. Roll window forward by one day

No look-ahead bias allowed.

---

## Feature Augmentation Framework

### Classical Baseline (required)

Classical augmented feature vector $\widetilde{\mathbf{X}}^{(c)} = \phi_{c}(\mathbf{X})$, including original features $\mathbf{X}$:

- Polynomial terms: $\{ X_{i},\; X_{i}^{2},\; X_{i}^{3} \}$
- Interaction terms: $\{ X_{i} X_{j} \mid i \neq j \}$
- Log/absolute transforms: $\log x, \; |x|$

### Quantum Feature Generation (required)

Quantum features via parameterized quantum feature map:

$$
\phi_{q}(\mathbf{X}) = \langle \psi(\mathbf{X}) | \hat{O} | \psi(\mathbf{X}) \rangle
$$

where $|\psi(\mathbf{X})\rangle = U(\mathbf{X}) |0\rangle^{\otimes n}$, $U(\mathbf{X})$ is a unitary parameterized by classical input, and $\hat{O}$ is a measurement operator (e.g., Pauli $Z_k$, pairwise $Z_k Z_l$, or higher-weight Pauli strings).

Angle encoding circuit:

$$
U(\mathbf{X}) = W_{\text{ent}} \prod_{k=1}^{n} R_Z^{(k)}(X_k)
$$

Augmented feature vector: $\widetilde{\mathbf{X}}^{(q)} = [\mathbf{X},\; \phi_{q}(\mathbf{X})]$

Encoding approaches to explore:
1. **Angle Encoding** — single-qubit rotations + entangling layers
2. **ZZ Feature Map** — rotations + ZZ interactions (Havlíček et al., *Nature* 2019)
3. **IQP Encoding** — diagonal circuit with Hadamards + phase gates

### Dimensionality Control

- Ridge: $\min_{\beta} \| Y - X\beta \|^{2} + \lambda \| \beta \|^{2}$
- Lasso: $\min_{\beta} \| Y - X\beta \|^{2} + \lambda \| \beta \|_{1}$

Same regularization strategy must be applied consistently across classical and quantum features.

### Model Requirements

Linear regression baseline is mandatory:

$$
\hat{Y} = \beta_{0} + \sum_{j} \beta_{j} \widetilde{X}_{j}
$$

- **Same model class** must be used for classical and quantum features (isolates feature effect)
- Optional extensions (kernels, trees) allowed but must clearly distinguish feature vs model effects

---

## Evaluation Expectations

- Aggregate out-of-sample MSE / MAE
- Information coefficient (correlation)
- Performance stability across time
- Feature count vs performance tradeoff
- Overfitting diagnostics
- **Quantum resource usage** (memory, circuit depth, qubit count) — required
- **Cost-performance tradeoffs** — part of result interpretation

## What Makes a Strong Submission

- Clean experimental design
- Apples-to-apples comparisons
- Honest reporting of negative results
- Careful control of overfitting
- Clear discussion of why quantum features help or fail
- Financial realism in evaluation

---

## AWS Resources

- **Workshop Studio**: https://catalog.workshops.aws (access code: `2447-0a1960-dc`)
- **Braket examples**: https://github.com/amazon-braket/amazon-braket-examples
- **Local simulator docs**: https://docs.aws.amazon.com/braket/latest/developerguide/braket-send-to-local-simulator.html
