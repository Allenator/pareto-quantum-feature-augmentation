"""Classical feature augmenters: polynomial, log/abs, RFF."""

import time

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from src.synthetic.augmenters.base import AugmenterResult, _make_result


class IdentityAugmenter:
    """No augmentation — raw features only."""
    name = "identity"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> AugmenterResult:
        return _make_result(X, None, self.name, 0.0)


class PolynomialAugmenter:
    """Polynomial feature expansion."""

    def __init__(self, degree: int = 2, interaction_only: bool = False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.name = f"poly_deg{degree}" + ("_interact" if interaction_only else "")
        self._poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=False,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._poly.fit(X_train)

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        X_poly = self._poly.transform(X)
        # PolynomialFeatures includes original features; extract only new ones
        X_new = X_poly[:, X.shape[1]:]
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)


class LogAbsAugmenter:
    """Log/absolute transforms + interaction terms."""

    def __init__(self):
        self.name = "interaction_log"
        self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._poly.fit(X_train)

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        # Interaction terms (no squares)
        X_interact = self._poly.transform(X)[:, X.shape[1]:]
        # log(|x| + 1) and |x| for each feature
        X_log = np.log(np.abs(X) + 1)
        X_abs = np.abs(X)
        X_new = np.hstack([X_interact, X_log, X_abs])
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)


class RFFAugmenter:
    """Random Fourier Features (RBF kernel approximation)."""

    def __init__(self, n_components: int = 10, gamma: float = 0.5, random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.name = f"rff_{n_components}"
        self._rff = RBFSampler(
            n_components=n_components, gamma=gamma, random_state=random_state,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._rff.fit(X_train)

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        X_new = self._rff.transform(X)
        elapsed = time.perf_counter() - t0
        n_random = self.n_components * X.shape[1]  # random projection matrix W
        return _make_result(X, X_new, self.name, elapsed, n_random_params=n_random)


class OracleAugmenter:
    """Oracle augmenter: adds the exact DGP terms X1*X3 and log(|X2|+1).

    This is the theoretical best a linear model can do on this DGP.
    """
    name = "oracle"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        X_new = np.column_stack([x1 * x3, np.log(np.abs(x2) + 1)])
        elapsed = time.perf_counter() - t0
        return _make_result(X, X_new, self.name, elapsed)
