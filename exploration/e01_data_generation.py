"""
Section 3: Synthetic Regime-Switching Data Generation

Generates the two-regime process defined in the challenge spec.
Regime 1 (75%): Y = 2*X1 - X2 + eps
Regime 2 (25%): Y = (X1*X3) + log(|X2| + 1) + eps
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

N_TRAIN = 10_000
N_TEST = 10_000
N_TOTAL = N_TRAIN + N_TEST
REGIME1_PROB = 0.75


def generate_regime_data(n_samples, seed=SEED):
    """Generate synthetic regime-switching data per challenge spec."""
    rng = np.random.default_rng(seed)

    regimes = rng.choice([1, 2], size=n_samples, p=[REGIME1_PROB, 1 - REGIME1_PROB])

    X1 = np.empty(n_samples)
    X2 = np.empty(n_samples)
    X3 = np.empty(n_samples)
    X4 = np.empty(n_samples)
    eps = rng.standard_normal(n_samples)

    r1 = regimes == 1
    r2 = regimes == 2
    n1, n2 = r1.sum(), r2.sum()

    # Regime 1 distributions
    X1[r1] = rng.standard_normal(n1)
    X2[r1] = rng.standard_normal(n1)
    X3[r1] = rng.standard_normal(n1)
    X4[r1] = rng.uniform(-1, 1, n1)

    # Regime 2 distributions
    mean_r2 = [3, 3]
    cov_r2 = [[1, 0.8], [0.8, 1]]
    x1x3_r2 = rng.multivariate_normal(mean_r2, cov_r2, n2)
    X1[r2] = x1x3_r2[:, 0]
    X3[r2] = x1x3_r2[:, 1]
    X2[r2] = rng.standard_cauchy(n2)
    X4[r2] = rng.exponential(1.0, n2)

    # Target variable
    Y = np.empty(n_samples)
    Y[r1] = 2 * X1[r1] - X2[r1] + eps[r1]
    Y[r2] = (X1[r2] * X3[r2]) + np.log(np.abs(X2[r2]) + 1) + eps[r2]

    X = np.column_stack([X1, X2, X3, X4])
    return X, Y, regimes


if __name__ == "__main__":
    X, Y, regimes = generate_regime_data(N_TOTAL)

    X_train, X_test, y_train, y_test, reg_train, reg_test = train_test_split(
        X, Y, regimes, train_size=N_TRAIN, shuffle=False, random_state=SEED
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Regime 1 fraction (train): {(reg_train == 1).mean():.2%}")
    print(f"Regime 1 fraction (test):  {(reg_test == 1).mean():.2%}")
    print(f"Y train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"Y test  mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
