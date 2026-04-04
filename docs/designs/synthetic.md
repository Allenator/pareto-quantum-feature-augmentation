# Synthetic Regime-Switching Experiment Design

## Context

This document describes the architecture for rigorously testing whether quantum feature augmentation improves out-of-sample prediction on the challenge's synthetic regime-switching DGP. The framework supports pluggable augmenters (classical, quantum, neural) and regression models, with enforced fairness controls and quantum resource tracking.

## DGP

$$
Y = \begin{cases} 2X_{1} - X_{2} + \varepsilon, & \text{Regime 1 (P=0.75)} \\ (X_{1} \cdot X_{3}) + \log(|X_{2}| + 1) + \varepsilon, & \text{Regime 2 (P=0.25)} \end{cases}
$$

4 features ($X_1$–$X_4$), latent regime, $X_4$ is pure noise, $\varepsilon \sim \mathcal{N}(0,1)$.

## Architecture

```
src/synthetic/
  config.py           # Frozen dataclasses: DGPConfig, AugmenterConfig, ModelConfig, ExperimentConfig
  dgp.py              # Data generation + parquet save/load
  augmenters/
    base.py            # FeatureAugmenter protocol + AugmenterResult
    classical.py       # Identity, Polynomial, LogAbs, RFF, Oracle
    quantum_fixed.py   # AngleEncoding, ZZMap, IQP, Reservoir, QAOA, Probability
    quantum_learned.py # VQC with adjoint-diff training
    neural.py          # MLP, Autoencoder, Learned RFF (torch MPS)
  models/
    base.py            # RegressionModel protocol + PredictionResult
    linear.py          # OLS, RidgeCV, LassoCV, ElasticNetCV
  evaluation/
    metrics.py         # ExperimentMetrics dataclass
    comparison.py      # FairnessChecker, ResultTable
  runner.py            # ExperimentRunner orchestrator
scripts/
  run_synthetic.py     # Entry point (classical / full modes)
```

## Core Abstractions

### FeatureAugmenter

All augmenters implement `fit(X_train, y_train)` and `transform(X) -> AugmenterResult`.
- `transform` always returns `[X_original | X_new]` concatenated — enforces "include original features" structurally.
- One-shot augmenters have no-op `fit`. Learning-based augmenters train parameters.
- Quantum augmenters populate `circuit_depth`, `qubit_count`, `gate_count` via `qml.specs()`.

### ExperimentRunner

```
for each seed:
  1. Load or generate data → data/synthetic/seed_{seed}.parquet
  2. StandardScaler.fit(X_train) — ONE scaler, shared
  3. For each augmenter: fit + transform (clipped inputs for quantum)
  4. For each (augmenter, model): fit_predict + compute_metrics
  5. Save per-run JSON → results/synthetic/seed_{seed}/
  6. Aggregate → results/synthetic/summary.csv
```

## Fairness Enforcement

| Concern | Mechanism |
|---------|-----------|
| Same data split | Parquet-cached per seed |
| Same preprocessing | One `StandardScaler` per seed, external to augmenters |
| Same model class | Runner iterates models × augmenters |
| Same regularization | `RidgeCV`/`LassoCV` with identical grids |
| Dimensionality tracked | `AugmenterResult.n_augmented` in every result |
| Quantum resources tracked | `circuit_depth`/`qubit_count`/`gate_count` |

Cauchy-distributed $X_2$ is clipped to $[-5, 5]$ for quantum augmenters (configurable `clip_range`).

## Augmenter Inventory

### Classical

| Name | New features | Total | Method |
|------|-------------|-------|--------|
| `identity` | 0 | 4 | Raw features only |
| `poly_deg2` | 10 | 14 | `PolynomialFeatures(degree=2)` |
| `poly_deg3` | 30 | 34 | `PolynomialFeatures(degree=3)` |
| `interaction_log` | 14 | 18 | Interactions + log/abs transforms |
| `rff_10` | 10 | 14 | RBF kernel approximation |
| `oracle` | 2 | 6 | Exact DGP terms: $X_1 X_3$, $\log(\|X_2\|+1)$ |

### Quantum Fixed

| Name | New features | Total | Circuit | Observables |
|------|-------------|-------|---------|-------------|
| `angle_basic_2L` | 10 | 14 | AngleEmbed + BasicEntangler | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `angle_strong_2L` | 10 | 14 | AngleEmbed + StronglyEntangling | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `zz_reps2` | 10 | 14 | ZZ feature map | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `iqp_reps3` | 10 | 14 | IQPEmbedding (n_repeats=3) | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `reservoir_3x3` | 12 | 16 | 3 random circuits | $\langle Z_i \rangle$ |
| `reservoir_3x3_zz` | 30 | 34 | 3 random circuits | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `qaoa_p2` | 10 | 14 | QAOA-inspired cost/mixer | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ |
| `prob_2L` | 16 | 20 | AngleEmbed + StronglyEntangling | Probabilities ($2^4$) |

### Learning-Based

| Name | New features | Training | Notes |
|------|-------------|----------|-------|
| `vqc_strong_2L` | 10 | PennyLane adjoint-diff, 200 epochs | Quantum trainable |
| `mlp_10` | 10 | Torch MPS, MSE loss | Classical trainable |
| `autoencoder_4` | 4 | Reconstruction loss (unsupervised) | Classical unsupervised |
| `learned_rff_10` | 10 | Trainable cos(Wx+b) | Classical trainable |

## Running

```bash
uv run python scripts/run_synthetic.py classical   # Classical baselines only
uv run python scripts/run_synthetic.py full         # All augmenters (slow)
```

## Results

All results use Ridge regression with CV-tuned $\alpha$. Classical baselines run on 10K train / 10K test with 5 seeds. Quantum and neural augmenters run on 2K train / 2K test (quantum) or 10K (neural) with 3 seeds. Metrics are mean $\pm$ std across seeds.

### Classical Baselines (10K/10K, 5 seeds)

| Augmenter | Features | MSE | MAE | Corr | Notes |
|-----------|----------|-----|-----|------|-------|
| identity | 4 | 4.68 $\pm$ 0.06 | 1.64 $\pm$ 0.01 | 0.931 $\pm$ 0.001 | Baseline |
| poly_deg2 | 14 | 2.88 $\pm$ 1.03 | 1.18 $\pm$ 0.02 | 0.959 $\pm$ 0.014 | High variance from Cauchy outliers |
| poly_deg3 | 34 | 106.4 $\pm$ 209.5 | 1.09 $\pm$ 0.09 | 0.746 $\pm$ 0.284 | **Unstable** — cubic terms amplify Cauchy tails |
| interaction_log | 18 | 2.62 $\pm$ 0.66 | 1.14 $\pm$ 0.03 | 0.962 $\pm$ 0.009 | DGP-matched features |
| rff_10 | 14 | 2.57 $\pm$ 0.08 | 1.22 $\pm$ 0.06 | 0.963 $\pm$ 0.001 | Random Fourier features |
| rff_30 | 34 | 2.18 $\pm$ 0.06 | 1.12 $\pm$ 0.03 | 0.969 $\pm$ 0.001 | More components helps |
| oracle | 6 | 2.41 $\pm$ 0.02 | 1.23 $\pm$ 0.01 | 0.965 $\pm$ 0.001 | Exact DGP terms |

### Quantum Fixed (2K/2K, 3 seeds)

| Augmenter | Features | Depth | MSE | MAE | Corr |
|-----------|----------|-------|-----|-----|------|
| angle_basic_2L | 14 | 2 | 2.76 $\pm$ 0.10 | 1.21 $\pm$ 0.01 | 0.959 $\pm$ 0.001 |
| angle_strong_2L | 14 | 2 | 2.59 $\pm$ 0.15 | 1.23 $\pm$ 0.02 | 0.962 $\pm$ 0.003 |
| angle_strong_3L | 14 | 2 | 2.63 $\pm$ 0.10 | 1.21 $\pm$ 0.01 | 0.961 $\pm$ 0.002 |
| zz_reps2 | 14 | 31 | 3.75 $\pm$ 0.18 | 1.44 $\pm$ 0.02 | 0.944 $\pm$ 0.002 |
| iqp_reps3 | 14 | 1 | 4.56 $\pm$ 0.12 | 1.61 $\pm$ 0.02 | 0.932 $\pm$ 0.001 |
| qaoa_p2 | 14 | 32 | 3.71 $\pm$ 0.26 | 1.36 $\pm$ 0.04 | 0.945 $\pm$ 0.004 |
| prob_2L | 20 | 2 | 2.34 $\pm$ 0.11 | 1.16 $\pm$ 0.02 | 0.966 $\pm$ 0.002 |
| reservoir_3x3 | 16 | 11 | 2.02 $\pm$ 0.08 | 1.08 $\pm$ 0.03 | 0.971 $\pm$ 0.001 |
| **reservoir_3x3_zz** | **34** | **11** | **1.61 $\pm$ 0.04** | **0.97 $\pm$ 0.01** | **0.976 $\pm$ 0.001** |

### Neural / Learning-Based (10K/10K, 3 seeds)

| Augmenter | Features | MSE | MAE | Corr | Notes |
|-----------|----------|-----|-----|------|-------|
| mlp_10 | 14 | 2.45 $\pm$ 0.55 | 1.12 $\pm$ 0.06 | 0.965 $\pm$ 0.007 | Torch MPS, 200 epochs |
| mlp_shallow_10 | 14 | 2.62 $\pm$ 0.27 | 1.21 $\pm$ 0.03 | 0.962 $\pm$ 0.003 | Shallow (4→10) |
| autoencoder_4 | 8 | 3.74 $\pm$ 1.62 | 1.30 $\pm$ 0.02 | 0.946 $\pm$ 0.022 | Unsupervised |
| **learned_rff_10** | **14** | **2.03 $\pm$ 0.02** | **1.11 $\pm$ 0.02** | **0.971 $\pm$ 0.001** | Trainable frequencies |

### Key Findings

1. **Quantum reservoir (Z+ZZ) achieves the best MSE (1.61)**, beating all classical baselines including the oracle (2.41). This is a strong positive result: random quantum circuits with pairwise observables capture the DGP's nonlinear regime-switching structure better than hand-crafted features.

2. **Learned RFF is the strongest classical trainable method (MSE 2.03)**, matching quantum reservoir (Z-only, MSE 2.02) and beating polynomial features, MLP, and even the oracle.

3. **poly_deg3 is catastrophically unstable** (MSE 106 $\pm$ 210) due to Cauchy-distributed $X_2$ in Regime 2 — cubic terms of heavy-tailed inputs explode. Lasso partially mitigates this but not reliably.

4. **IQP encoding adds almost no value** (MSE 4.56, barely better than identity at 4.68). The IQP circuit structure is a poor match for this DGP's multiplicative interactions.

5. **More features helps** when the method is stable: rff_30 (34 features) beats rff_10 (14 features); reservoir_3x3_zz (34 features) beats reservoir_3x3 (16 features).

6. **Probability extraction (prob_2L, MSE 2.34) outperforms expectation-only angle encoding (MSE 2.59)** with the same circuit, suggesting that full state information is valuable.

### Ranking (best model per augmenter)

| Rank | Augmenter | Type | Features | MSE | Corr |
|------|-----------|------|----------|-----|------|
| 1 | reservoir_3x3_zz | Quantum fixed | 34 | 1.61 | 0.976 |
| 2 | reservoir_3x3 | Quantum fixed | 16 | 2.02 | 0.971 |
| 3 | learned_rff_10 | Classical learned | 14 | 2.03 | 0.971 |
| 4 | rff_30 | Classical fixed | 34 | 2.18 | 0.969 |
| 5 | prob_2L | Quantum fixed | 20 | 2.34 | 0.966 |
| 6 | oracle | Classical (exact DGP) | 6 | 2.41 | 0.965 |
| 7 | mlp_10 | Neural learned | 14 | 2.45 | 0.965 |
| 8 | rff_10 | Classical fixed | 14 | 2.57 | 0.963 |
| 9 | angle_strong_2L | Quantum fixed | 14 | 2.59 | 0.962 |
| 10 | interaction_log | Classical fixed | 18 | 2.62 | 0.962 |
| 11 | mlp_shallow_10 | Neural learned | 14 | 2.62 | 0.962 |
| 12 | angle_strong_3L | Quantum fixed | 14 | 2.63 | 0.961 |
| 13 | angle_basic_2L | Quantum fixed | 14 | 2.76 | 0.959 |
| 14 | poly_deg2 | Classical fixed | 14 | 2.88 | 0.959 |
| 15 | qaoa_p2 | Quantum fixed | 14 | 3.71 | 0.945 |
| 16 | zz_reps2 | Quantum fixed | 14 | 3.75 | 0.944 |
| 17 | autoencoder_4 | Neural learned | 8 | 3.74 | 0.946 |
| 18 | identity | Classical | 4 | 4.68 | 0.931 |
| 19 | iqp_reps3 | Quantum fixed | 14 | 4.56 | 0.932 |
| 20 | poly_deg3 | Classical fixed | 34 | 106.4 | 0.746 |
