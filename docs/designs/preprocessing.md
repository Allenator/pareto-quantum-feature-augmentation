# Preprocessing Pipeline

## Overview

The preprocessing pipeline transforms raw DGP outputs into augmented feature matrices ready for regression. Every step is applied identically to all augmenters (classical and quantum) within each seed, enforcing fair comparison. The pipeline is implemented in `src/synthetic/runner.py`.

## Pipeline

```
Raw DGP data → Train/test split → StandardScaler → [Clip for quantum] → Feature augmentation → Ridge/Lasso
```

### Step 1: Data Generation

The synthetic regime-switching DGP (`src/synthetic/dgp.py`) generates raw features and targets:

$$
Y = \begin{cases} 2X_1 - X_2 + \varepsilon & \text{Regime 1 (P=0.75)} \\ X_1 X_3 + \log(|X_2|+1) + \varepsilon & \text{Regime 2 (P=0.25)} \end{cases}
$$

**Feature distributions:**

| Feature | Regime 1 | Regime 2 |
|---------|----------|----------|
| $X_1$ | $\mathcal{N}(0,1)$ | $\mathcal{N}(3,1)$, correlated with $X_3$ ($\rho=0.8$) |
| $X_2$ | $\mathcal{N}(0,1)$ | $\text{Cauchy}(0,1)$ — heavy tails, infinite variance |
| $X_3$ | $\mathcal{N}(0,1)$ | $\mathcal{N}(3,1)$, correlated with $X_1$ |
| $X_4$ | $\text{Uniform}(-1,1)$ | $\text{Exp}(\lambda=1)$ |
| $\varepsilon$ | $\mathcal{N}(0,1)$ | $\mathcal{N}(0,1)$ |

Data is generated once per seed and cached as parquet in `data/synthetic/seed_{seed}.parquet` (git LFS tracked). Each parquet contains columns `X1, X2, X3, X4, Y, regime, split`.

**Configuration** (`DGPConfig`):
- `n_train`: 10,000 (default) or 2,000 (unified factorial)
- `n_test`: 10,000 or 2,000
- `regime1_prob`: 0.75
- `seed`: varies (42, 123, 456, 789, 1024)

### Step 2: Train/Test Split

The first `n_train` samples are training, the remaining `n_test` are test. No shuffling — the split is sequential, preserving any temporal structure. This is consistent across all augmenters and seeds.

```python
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
```

### Step 3: Standardization

A single `StandardScaler` is fit on training data and applied to both train and test:

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit on train
X_test_s = scaler.transform(X_test)        # transform test (no re-fit)
```

This is done **once per seed** before any augmenter sees the data. All augmenters receive identically scaled inputs.

**Why StandardScaler**: The DGP's features have different scales and distributions (Gaussian, Cauchy, Uniform, Exponential). Standardization centers each feature to mean 0, std 1, which is critical for:
- Ridge/Lasso regularization to penalize coefficients fairly across features
- Quantum circuits where rotation angles $R_Y(x_i)$ are sensitive to input scale

**Cauchy tail behavior**: $X_2$ in Regime 2 follows a Cauchy distribution with infinite variance. After `StandardScaler`, extreme Cauchy values (e.g., $|X_2| > 100$) are standardized using the sample mean/std, which are themselves unstable. This causes:
- Standardized $X_2$ values that can be very large (10+)
- Polynomial features of these values explode (causing the poly_deg3 instability at MSE 106)
- Quantum rotation angles wrap around ($R_Y(10)$ wraps ~1.6 full turns)

### Step 4: Clipping (Quantum Only)

For quantum augmenters, standardized features are clipped to $[-c, c]$ where $c$ is the `clip_range` parameter (default 5.0):

```python
if aug_config.kind.startswith("quantum") and clip_range is not None:
    X_tr = np.clip(X_train_s, -clip_range, clip_range)
    X_te = np.clip(X_test_s, -clip_range, clip_range)
```

**Why clip**: Quantum rotation gates are periodic ($R_Y$ has period $4\pi \approx 12.6$). Without clipping, extreme Cauchy values map to nearly identical rotations (wrapping), destroying discriminative information. Clipping to $[-5, 5]$ keeps rotation angles within $\sim 0.8$ turns, where trigonometric features are most informative.

**Classical augmenters receive unclipped data** — they can handle large values natively (polynomial features benefit from the full range).

**Impact**: Clipping affects ~0.5% of samples (only extreme Cauchy tails in Regime 2). For Regime 1, all features are $\mathcal{N}(0,1)$ after standardization, so clipping at $\pm 5$ has negligible effect.

### Step 5: Feature Augmentation

Each augmenter receives the preprocessed data and produces an augmented feature matrix:

```python
augmenter.fit(X_train_preprocessed, y_train)
train_result = augmenter.transform(X_train_preprocessed)  # (n_train, n_original + n_augmented)
test_result = augmenter.transform(X_test_preprocessed)     # (n_test, n_original + n_augmented)
```

The augmented matrix always contains the original 4 features as the first columns, followed by the new augmented features. This is enforced structurally inside each augmenter's `transform` method via `_make_result`.

### Step 6: Regression

The augmented feature matrix is passed to Ridge or Lasso regression with cross-validated regularization:

```python
model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
model.fit(X_train_augmented, y_train)
y_pred = model.predict(X_test_augmented)
```

The same `alpha_grid` and `cv_folds` are used for all augmenters — no per-augmenter tuning of the downstream model.

## Fairness Controls

| Concern | Enforcement |
|---------|-------------|
| Same data split | Parquet-cached per seed, sequential split |
| Same scaling | One `StandardScaler` per seed, external to augmenters |
| Same regularization | Identical `alpha_grid` and `cv_folds` for all |
| Same model class | All augmenters evaluated with the same Ridge/Lasso |
| Original features included | Enforced inside `_make_result` — first 4 columns are always raw features |
| Clipping only for quantum | Classical augmenters receive unclipped standardized data |

## Configuration Parameters

| Parameter | Default | Location | Effect |
|-----------|---------|----------|--------|
| `n_train` | 10,000 | `DGPConfig` | Training set size |
| `n_test` | 10,000 | `DGPConfig` | Test set size |
| `clip_range` | 5.0 | `ExperimentConfig` | Clipping bound for quantum augmenters (None = no clip) |
| `alpha_grid` | (0.001, 0.01, 0.1, 1.0, 10.0) | `ModelConfig` | Ridge/Lasso CV search grid |
| `cv_folds` | 5 | `ModelConfig` | Number of CV folds for regularization tuning |
| `seeds` | [42, 123, 456, 789, 1024] | `ExperimentConfig` | Random seeds for data generation |
