"""Data generating process: synthetic regime-switching model."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.synthetic.config import DGPConfig


def generate_regime_data(config: DGPConfig) -> pd.DataFrame:
    """Generate synthetic regime-switching data per challenge spec.

    Returns DataFrame with columns: X1, X2, X3, X4, Y, regime, split.
    """
    n_total = config.n_train + config.n_test
    rng = np.random.default_rng(config.seed)

    regimes = rng.choice(
        [1, 2], size=n_total, p=[config.regime1_prob, 1 - config.regime1_prob]
    )

    X1 = np.empty(n_total)
    X2 = np.empty(n_total)
    X3 = np.empty(n_total)
    X4 = np.empty(n_total)
    eps = rng.standard_normal(n_total)

    r1 = regimes == 1
    r2 = regimes == 2
    n1, n2 = r1.sum(), r2.sum()

    # Regime 1
    X1[r1] = rng.standard_normal(n1)
    X2[r1] = rng.standard_normal(n1)
    X3[r1] = rng.standard_normal(n1)
    X4[r1] = rng.uniform(-1, 1, n1)

    # Regime 2
    x1x3 = rng.multivariate_normal([3, 3], [[1, 0.8], [0.8, 1]], n2)
    X1[r2] = x1x3[:, 0]
    X3[r2] = x1x3[:, 1]
    X2[r2] = rng.standard_cauchy(n2)
    X4[r2] = rng.exponential(1.0, n2)

    # Target
    Y = np.empty(n_total)
    Y[r1] = 2 * X1[r1] - X2[r1] + eps[r1]
    Y[r2] = (X1[r2] * X3[r2]) + np.log(np.abs(X2[r2]) + 1) + eps[r2]

    split = np.array(["train"] * config.n_train + ["test"] * config.n_test)

    return pd.DataFrame({
        "X1": X1, "X2": X2, "X3": X3, "X4": X4,
        "Y": Y, "regime": regimes, "split": split,
    })


def save_data(df: pd.DataFrame, data_dir: str, seed: int) -> Path:
    """Save generated data to parquet."""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"seed_{seed}.parquet"
    df.to_parquet(filepath, index=False)
    return filepath


def load_data(data_dir: str, seed: int) -> pd.DataFrame | None:
    """Load data from parquet if it exists."""
    filepath = Path(data_dir) / f"seed_{seed}.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None


def get_or_generate(config: DGPConfig, data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached data or generate and save.

    Returns (X_train, X_test, y_train, y_test, regime_train, regime_test).
    """
    df = load_data(data_dir, config.seed)
    if df is None:
        df = generate_regime_data(config)
        save_data(df, data_dir, config.seed)

    feature_cols = ["X1", "X2", "X3", "X4"]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    return (
        train[feature_cols].values,
        test[feature_cols].values,
        train["Y"].values,
        test["Y"].values,
        train["regime"].values,
        test["regime"].values,
    )
