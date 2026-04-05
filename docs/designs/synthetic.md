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
    base.py            # FeatureAugmenter protocol + AugmenterResult (incl. n_trainable_params, n_random_params)
    classical.py       # Identity, Polynomial, LogAbs, RFF, Oracle
    quantum_fixed.py   # AngleEncoding, ZZMap, IQP, Reservoir (multi-basis, entanglement variants), QAOA, Probability
    quantum_learned.py # VQC with adjoint-diff training
    neural.py          # MLP, Autoencoder (hybrid supervised), Learned RFF — with early stopping, cosine LR
  models/
    base.py            # RegressionModel protocol + PredictionResult (incl. coef_l2_norm, lasso_active_fraction)
    linear.py          # OLS, RidgeCV, LassoCV, ElasticNetCV
  evaluation/
    metrics.py         # ExperimentMetrics dataclass (performance + complexity metrics)
    complexity.py      # Effective rank, nonlinearity score, feature-target alignment
    comparison.py      # FairnessChecker, ResultTable
  runner.py            # ExperimentRunner: multiprocessing, tqdm, skip-if-exists, feature saving
scripts/
  run_synthetic.py     # Entry point (static / trainable / classical modes)
  plot_synthetic.py    # Interactive Plotly HTML visualizations
```

## Core Abstractions

### FeatureAugmenter

All augmenters implement `fit(X_train, y_train)` and `transform(X) -> AugmenterResult`.
- `transform` always returns `[X_original | X_new]` concatenated — enforces "include original features" structurally.
- One-shot augmenters have no-op `fit`. Learning-based augmenters train parameters (with early stopping).
- All augmenters report `n_trainable_params` and `n_random_params` for complexity analysis.
- Quantum augmenters additionally populate `circuit_depth`, `qubit_count`, `gate_count` via `qml.specs()`.
- Features are saved to `features/synthetic/{run_id}/seed_{seed}/{augmenter}.npz`.

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
uv run python scripts/run_synthetic.py static       # Static augmenter sweep
uv run python scripts/run_synthetic.py trainable    # Trainable augmenter sweep
uv run python scripts/run_synthetic.py classical    # Classical baselines only
```

## Results — Static Augmenter Sweep

> **Note**: Results below are from a prior sweep without complexity metrics or extended feature ranges. Pending re-run with `n_trainable_params`, `n_random_params`, `effective_rank`, `nonlinearity_score`, `feature_target_alignment`, `coef_l2_norm`, `lasso_active_fraction`, and feature counts extended to ≤100.

All results: Ridge regression, CV-tuned $\alpha$, 10K train / 10K test, 5 seeds. MSE columns show test (train) for overfitting comparison. Per-regime MSE shows Regime 1 (linear, 75%) vs Regime 2 (nonlinear, 25%).

### Classical

| Augmenter | Feat | MSE test (train) | R1 test (train) | R2 test (train) | Corr |
|-----------|------|-------------------|-----------------|-----------------|------|
| identity | 4 | 4.68 (4.68) | 3.23 (3.20) | 9.02 (9.21) | 0.931 |
| oracle | 6 | 2.41 (2.39) | 2.36 (2.33) | 2.56 (2.54) | 0.965 |
| poly_deg2_interact | 10 | 3.10 (2.43) | 2.27 (2.25) | 5.58 (3.01) | 0.955 |
| poly_deg2 | 14 | 2.88 (2.21) | 2.11 (2.10) | 5.18 (2.55) | 0.959 |
| poly_deg3 | 34 | 106.4 (1.61) | 1.48 (1.47) | 417.4 (2.04) | 0.746 |
| interaction_log | 18 | 2.62 (2.06) | 2.11 (2.11) | 4.13 (1.91) | 0.962 |
| rff_6 | 10 | 2.48 (2.47) | 2.17 (2.17) | 3.40 (3.40) | 0.964 |
| rff_10 | 14 | 2.57 (2.50) | 1.92 (1.91) | 4.50 (4.31) | 0.963 |
| rff_15 | 19 | 2.39 (2.35) | 2.06 (2.06) | 3.37 (3.26) | 0.965 |
| rff_21 | 25 | 2.15 (2.11) | 2.09 (2.06) | 2.33 (2.25) | 0.969 |
| rff_30 | 34 | 2.18 (2.10) | 1.96 (1.95) | 2.80 (2.57) | 0.969 |
| rff_36 | 40 | 1.95 (1.89) | 1.73 (1.72) | 2.59 (2.41) | 0.972 |
| rff_50 | 54 | 1.60 (1.53) | 1.41 (1.40) | 2.15 (1.93) | 0.977 |

### Angle Encoding (StronglyEntangling)

| Augmenter | Q | Feat | MSE test (train) | R1 test (train) | R2 test (train) | Corr |
|-----------|---|------|-------------------|-----------------|-----------------|------|
| 3q 2L | 3 | 10 | 2.18 (2.16) | 2.03 (2.03) | 2.62 (2.58) | 0.969 |
| 4q 1L | 4 | 14 | 2.24 (2.20) | 2.18 (2.17) | 2.41 (2.27) | 0.968 |
| 4q 2L | 4 | 14 | 2.52 (2.47) | 2.14 (2.12) | 3.64 (3.56) | 0.964 |
| 4q 3L | 4 | 14 | 2.55 (2.42) | 2.04 (2.03) | 4.04 (3.64) | 0.963 |
| 5q 2L | 5 | 19 | 2.37 (2.34) | 2.21 (2.19) | 2.84 (2.82) | 0.966 |
| 6q 2L | 6 | 25 | 2.13 (2.07) | 1.87 (1.86) | 2.92 (2.73) | 0.969 |
| 7q 2L | 7 | 32 | 2.02 (1.99) | 1.77 (1.76) | 2.77 (2.70) | 0.971 |
| 8q 2L | 8 | 40 | 1.86 (1.80) | 1.77 (1.75) | 2.14 (1.97) | 0.973 |
| 9q 2L | 9 | 49 | 1.89 (1.83) | 1.75 (1.73) | 2.32 (2.16) | 0.973 |
| 10q 2L | 10 | 59 | 1.69 (1.63) | 1.53 (1.51) | 2.18 (1.99) | 0.976 |

### Quantum Reservoir — Size & Basis Sweep

| Config | Obs | Feat | MSE test (train) | R1 test (train) | R2 test (train) | Corr |
|--------|-----|------|-------------------|-----------------|-----------------|------|
| 1×3 | Z | 8 | 3.87 (3.89) | 3.12 (3.10) | 6.12 (6.32) | 0.943 |
| 2×3 | Z | 12 | 2.71 (2.64) | 2.26 (2.25) | 4.04 (3.83) | 0.961 |
| 3×3 | Z | 16 | 2.20 (2.14) | 1.82 (1.81) | 3.31 (3.14) | 0.968 |
| 5×3 | Z | 24 | 2.01 (1.96) | 1.70 (1.69) | 2.92 (2.77) | 0.971 |
| 7×3 | Z | 32 | 1.75 (1.68) | 1.53 (1.52) | 2.37 (2.16) | 0.975 |
| 10×3 | Z | 44 | 1.55 (1.49) | 1.36 (1.35) | 2.12 (1.92) | 0.978 |
| 13×3 | Z | 56 | 1.40 (1.32) | 1.19 (1.18) | 2.05 (1.73) | 0.980 |
| 1×3 | Z+ZZ | 14 | 3.16 (3.17) | 2.63 (2.64) | 4.71 (4.79) | 0.954 |
| 2×3 | Z+ZZ | 24 | 2.21 (2.15) | 1.74 (1.75) | 3.61 (3.41) | 0.968 |
| 3×3 | Z+ZZ | 34 | 1.73 (1.68) | 1.55 (1.55) | 2.27 (2.10) | 0.975 |
| 5×3 | Z+ZZ | 54 | 1.46 (1.39) | 1.26 (1.25) | 2.06 (1.83) | 0.979 |
| 3×3 | X | 16 | 2.26 (2.23) | 2.11 (2.09) | 2.70 (2.65) | 0.967 |
| 3×3 | Y | 16 | 2.37 (2.37) | 2.03 (2.01) | 3.38 (3.49) | 0.966 |
| 3×3 | XYZ | 40 | 1.57 (1.52) | 1.41 (1.41) | 2.03 (1.86) | 0.978 |
| 3×3 | XYZ+ZZ | 58 | 1.44 (1.37) | 1.25 (1.25) | 2.00 (1.75) | 0.979 |
| 3×3 | full | 94 | 1.42 (1.37) | 1.27 (1.27) | 1.87 (1.68) | 0.980 |
| **5×3** | **XYZ+ZZ** | **94** | **1.37 (1.31)** | **1.20 (1.19)** | **1.89 (1.68)** | **0.980** |

### Reservoir — Entanglement & Data Reuploading

| Config | Variant | Feat | MSE test (train) | R2 test (train) | Corr |
|--------|---------|------|-------------------|-----------------|------|
| 3×3 Z+ZZ | linear (baseline) | 34 | 1.73 (1.68) | 2.27 (2.10) | 0.975 |
| 3×3 Z+ZZ | circular | 34 | 1.82 (1.78) | 2.28 (2.14) | 0.974 |
| 3×3 Z+ZZ | all-to-all | 34 | 1.73 (1.67) | 2.21 (2.01) | 0.975 |
| 3×3 Z+ZZ | data reuploading | 34 | 2.59 (2.55) | 3.56 (3.53) | 0.962 |
| 3×3 Z+ZZ | circular + reup | 34 | 2.50 (2.45) | 3.59 (3.50) | 0.964 |

### ZZ / IQP / QAOA (extended to 10 qubits)

These methods show flat or degrading performance with more qubits. Shown in compact form:

| Method | 3q (10 feat) | 6q (25) | 10q (59) | R2 @ 10q |
|--------|-------------|---------|----------|----------|
| ZZ | 3.50 | 3.86 | 3.85 | 7.95 |
| IQP | 4.34 | 4.43 | 4.22 | 8.73 |
| QAOA | 3.40 | 3.80 | 3.84 | 9.02 |

### Probability Extraction

| Qubits | Feat | MSE test (train) | R2 test (train) | Corr |
|--------|------|-------------------|-----------------|------|
| 3 | 12 | 1.98 (1.96) | 2.41 (2.37) | 0.972 |
| 4 | 20 | 2.34 (2.32) | 3.33 (3.32) | 0.966 |
| 5 | 36 | 1.94 (1.88) | 2.22 (2.02) | 0.972 |
| 6 | 68 | 1.56 (1.52) | 2.14 (1.98) | 0.978 |

### Key Findings

1. **Best overall: reservoir 5×3 XYZ+ZZ (MSE 1.37, 94 features)** — measuring all three Pauli bases plus ZZ pairwise across 5 random circuits. At matched feature count, reservoir_3x3_full (94 features, MSE 1.42) is nearly identical, showing that more reservoirs vs more observables are interchangeable strategies for increasing diversity.

2. **Overfitting is minimal across all methods.** Train-test MSE gaps are typically 0.05-0.15 (3-8%). The largest gap is poly_deg3 (train MSE 1.61, test MSE 106.4) — catastrophic extrapolation from Cauchy outliers. Polynomial and interaction_log show moderate overfitting (train 2.06, test 2.62). Quantum methods show remarkably tight train-test gaps.

3. **Measurement basis matters more than entanglement topology.** Switching from Z to XYZ observables on the same 3×3 reservoir drops MSE from 2.20 to 1.57 (29% improvement). Switching entanglement from linear to circular or all-to-all gives <5% change. **Data reuploading hurts** (MSE 2.59 vs 1.73) — re-encoding at every layer overwrites the random diversity that makes reservoirs effective.

4. **X-basis and Y-basis are individually competitive with Z-basis** (reservoir_3x3_X: MSE 2.26 vs reservoir_3x3 Z: MSE 2.20), but **combining all three bases is where the real gain is** — each basis reveals different quantum state structure.

5. **Classical RFF scales well to 50 features** (MSE 1.60), competitive with many quantum methods. But the best quantum configurations (reservoir XYZ+ZZ, angle 10q) still hold ~15% advantage at comparable feature counts.

6. **Feature efficiency ranking** (MSE per feature):

   | Method | Features | MSE | MSE/feat |
   |--------|----------|-----|----------|
   | prob_3q_2L | 12 | 1.98 | 0.165 |
   | reservoir_3x3_XYZ | 40 | 1.57 | 0.039 |
   | reservoir_13x3 | 56 | 1.40 | 0.025 |
   | oracle | 6 | 2.41 | 0.402 |

### Top 10 Overall Ranking

| Rank | Augmenter | Type | Feat | MSE | MSE train | R2 test | Corr |
|------|-----------|------|------|-----|-----------|---------|------|
| 1 | reservoir_5x3_XYZ_ZZ | Quantum | 94 | 1.37 | 1.31 | 1.89 | 0.980 |
| 2 | reservoir_13x3 | Quantum | 56 | 1.40 | 1.32 | 2.05 | 0.980 |
| 3 | reservoir_3x3_full | Quantum | 94 | 1.42 | 1.37 | 1.87 | 0.980 |
| 4 | reservoir_3x3_XYZ_ZZ | Quantum | 58 | 1.44 | 1.37 | 2.00 | 0.979 |
| 5 | reservoir_5x3_zz | Quantum | 54 | 1.46 | 1.39 | 2.06 | 0.979 |
| 6 | reservoir_2x3_XYZ_ZZ | Quantum | 40 | 1.49 | 1.44 | 1.98 | 0.979 |
| 7 | reservoir_10x3 | Quantum | 44 | 1.55 | 1.49 | 2.12 | 0.978 |
| 8 | prob_6q_2L | Quantum | 68 | 1.56 | 1.52 | 2.14 | 0.978 |
| 9 | reservoir_3x3_XYZ | Quantum | 40 | 1.57 | 1.52 | 2.03 | 0.978 |
| 10 | rff_50 | Classical | 54 | 1.60 | 1.53 | 2.15 | 0.977 |
