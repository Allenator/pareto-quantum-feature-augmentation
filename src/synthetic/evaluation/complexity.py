"""Feature-level complexity metrics computed from augmented feature matrices."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def compute_complexity_metrics(
    X_augmented: np.ndarray,
    y: np.ndarray,
    n_original: int,
) -> dict:
    """Compute complexity metrics from the augmented feature matrix.

    Args:
        X_augmented: Full feature matrix (n_samples, n_original + n_augmented)
        y: Target variable
        n_original: Number of original (raw) features (first columns)

    Returns:
        Dict with effective_rank, nonlinearity_score, feature_target_alignment.
    """
    n_aug = X_augmented.shape[1] - n_original
    if n_aug == 0:
        return {
            "effective_rank": 0.0,
            "nonlinearity_score": None,
            "feature_target_alignment": None,
        }

    X_raw = X_augmented[:, :n_original]
    X_new = X_augmented[:, n_original:]

    return {
        "effective_rank": _effective_rank(X_new),
        "nonlinearity_score": _nonlinearity_score(X_raw, X_new),
        "feature_target_alignment": _feature_target_alignment(X_new, y),
    }


def _effective_rank(X: np.ndarray) -> float:
    """Shannon entropy of normalized singular values."""
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return round(float(np.exp(entropy)), 4)


def _nonlinearity_score(X_raw: np.ndarray, X_new: np.ndarray) -> float:
    """Average (1 - R²) of linear fit from raw features to each augmented feature."""
    n_aug = X_new.shape[1]
    residuals = []
    for j in range(n_aug):
        lr = LinearRegression().fit(X_raw, X_new[:, j])
        r2 = lr.score(X_raw, X_new[:, j])
        residuals.append(1.0 - max(0.0, r2))
    return round(float(np.mean(residuals)), 6)


def _feature_target_alignment(X_new: np.ndarray, y: np.ndarray) -> float:
    """Average |correlation| between each augmented feature and target."""
    n_aug = X_new.shape[1]
    corrs = []
    for j in range(n_aug):
        col = X_new[:, j]
        if np.std(col) < 1e-10:
            corrs.append(0.0)
            continue
        r, _ = pearsonr(col, y)
        corrs.append(abs(r))
    return round(float(np.mean(corrs)), 6)
