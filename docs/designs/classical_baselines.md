# Classical Feature Augmentation Baselines

## Overview

Classical baselines serve two purposes: (1) establish performance floors/ceilings that quantum methods must beat to demonstrate value, and (2) provide feature-count-matched comparisons for fairness analysis. All baselines use the same Ridge regression (CV-tuned $\alpha$) downstream model as quantum methods.

## Methods

### Identity (no augmentation)

No feature augmentation. The 4 raw features $X_1, X_2, X_3, X_4$ are passed directly to Ridge regression.

- **Features**: 4
- **MSE**: 4.68 $\pm$ 0.06
- **Correlation**: 0.931
- **Role**: Lower bound — any augmentation should beat this

### Oracle (hand-crafted DGP terms)

Adds the two exact functional forms from the DGP: $X_1 X_3$ (Regime 2 interaction) and $\log(|X_2| + 1)$ (Regime 2 nonlinear transform). This is the theoretical best a linear model can do with 2 augmented features, since it directly contains the true signal components.

- **Features**: 6 (4 original + 2 augmented)
- **MSE**: 2.41 $\pm$ 0.02
- **Correlation**: 0.965
- **Role**: Performance ceiling for feature engineering — any method that beats this has discovered structure beyond the DGP's explicit functional forms

### Polynomial (degree 2)

`sklearn.preprocessing.PolynomialFeatures(degree=2)`: generates all monomials up to degree 2 — $X_i^2$ and $X_i X_j$ terms. Includes the critical $X_1 X_3$ interaction but also irrelevant terms like $X_4^2$.

| Variant | Features | MSE | Correlation |
|---------|----------|-----|-------------|
| Full (squares + interactions) | 14 | 2.88 $\pm$ 1.03 | 0.959 |
| Interaction-only (no squares) | 10 | 3.10 $\pm$ 0.88 | 0.955 |

**Notable**: High variance across seeds ($\pm$ 1.03) due to Cauchy-distributed $X_2$ in Regime 2. The squared terms of heavy-tailed inputs are unstable.

### Interaction + Log/Abs transforms

Combines interaction-only polynomial terms ($X_i X_j$, 6 features) with $\log(|X_i| + 1)$ and $|X_i|$ transforms (8 features). Hand-crafted to partially match the DGP's functional forms.

- **Features**: 18 (4 original + 14 augmented)
- **MSE**: 2.62 $\pm$ 0.66
- **Correlation**: 0.962
- **Role**: Tests whether domain-knowledge-informed classical transforms can match quantum methods

### Random Fourier Features (RFF)

`sklearn.kernel_approximation.RBFSampler`: approximates the RBF kernel $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$ via random cosine projections $\phi(\mathbf{x}) = \sqrt{2/D} \cos(\mathbf{w}^T \mathbf{x} + b)$ where $\mathbf{w} \sim \mathcal{N}(0, \gamma I)$ and $b \sim \text{Uniform}(0, 2\pi)$.

The primary classical competitor to quantum reservoir methods — both use random projections into a nonlinear feature space, differing only in whether the projection is classical (cosine) or quantum (circuit).

| Components | Features | Random params | MSE | Correlation |
|-----------|----------|---------------|-----|-------------|
| 6 | 10 | 24 | 2.48 $\pm$ 0.03 | 0.964 |
| 10 | 14 | 40 | 2.57 $\pm$ 0.08 | 0.963 |
| 15 | 19 | 60 | 2.39 $\pm$ 0.08 | 0.965 |
| 21 | 25 | 84 | 2.15 $\pm$ 0.10 | 0.969 |
| 30 | 34 | 120 | 2.18 $\pm$ 0.06 | 0.969 |
| 36 | 40 | 144 | 1.95 $\pm$ 0.15 | 0.972 |
| 50 | 54 | 200 | 1.60 $\pm$ 0.25 | 0.977 |
| 96 | 100 | 384 | 1.45 $\pm$ 0.26 | 0.979 |

RFF scales steadily with component count but has **increasing variance** at higher dimensions — the random projection quality becomes seed-dependent.

## Comparison Table

All methods, sorted by MSE (10K train / 10K test, 5 seeds, Ridge):

| Method | Type | Features | Random params | MSE | $\pm$ | Corr |
|--------|------|----------|---------------|-----|-------|------|
| RFF 96 | Random projection | 100 | 384 | 1.45 | 0.26 | 0.979 |
| RFF 50 | Random projection | 54 | 200 | 1.60 | 0.25 | 0.977 |
| RFF 36 | Random projection | 40 | 144 | 1.95 | 0.15 | 0.972 |
| RFF 21 | Random projection | 25 | 84 | 2.15 | 0.10 | 0.969 |
| RFF 30 | Random projection | 34 | 120 | 2.18 | 0.06 | 0.969 |
| RFF 15 | Random projection | 19 | 60 | 2.39 | 0.08 | 0.965 |
| Oracle | Hand-crafted | 6 | 0 | 2.41 | 0.02 | 0.965 |
| RFF 6 | Random projection | 10 | 24 | 2.48 | 0.03 | 0.964 |
| RFF 10 | Random projection | 14 | 40 | 2.57 | 0.08 | 0.963 |
| Interaction + Log/Abs | Hand-crafted | 18 | 0 | 2.62 | 0.66 | 0.962 |
| Polynomial (deg 2) | Deterministic | 14 | 0 | 2.88 | 1.03 | 0.959 |
| Poly (interact only) | Deterministic | 10 | 0 | 3.10 | 0.88 | 0.955 |
| Identity | None | 4 | 0 | 4.68 | 0.06 | 0.931 |

## Key Observations

1. **RFF is the strongest classical baseline**, approaching the noise floor at 100 features (MSE 1.45). It is the primary competitor to quantum reservoir methods.

2. **Deterministic methods (polynomial) are unstable** on this DGP due to Cauchy-distributed $X_2$. The variance across seeds ($\pm$ 1.03) is 4× larger than RFF at similar feature counts.

3. **The oracle achieves MSE 2.41 with only 6 features** — no other method matches this feature efficiency. It proves that the DGP's signal is recoverable with the right 2 features.

4. **RFF's variance increases with dimension** (0.03 at 10 features → 0.26 at 100 features), suggesting the random projection quality is seed-sensitive at scale. Quantum reservoir methods show lower variance at comparable feature counts.

5. **No classical method beats MSE 1.45** — the quantum Pareto frontier reaches MSE 1.38 at 94 features, a 5% improvement over the best classical baseline at matched feature count.
