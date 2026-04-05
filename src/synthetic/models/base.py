"""RegressionModel protocol and PredictionResult dataclass."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class PredictionResult:
    """Output of a regression model."""
    y_pred: np.ndarray
    y_train_pred: np.ndarray | None
    model_name: str
    chosen_alpha: float | None  # None for OLS
    n_features: int
    coef_l2_norm: float | None = None
    lasso_active_fraction: float | None = None


@runtime_checkable
class RegressionModel(Protocol):
    """Protocol for regression models."""
    name: str

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> PredictionResult:
        """Fit on training data (with CV for hyperparams) and predict on test."""
        ...
