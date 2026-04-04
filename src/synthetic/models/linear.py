"""Linear regression model wrappers with consistent CV tuning."""

import numpy as np
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    RidgeCV,
)

from src.synthetic.models.base import PredictionResult


class OLSModel:
    """Ordinary least squares (no regularization)."""
    name = "ols"

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> PredictionResult:
        model = LinearRegression().fit(X_train, y_train)
        return PredictionResult(
            y_pred=model.predict(X_test),
            model_name=self.name,
            chosen_alpha=None,
            n_features=X_train.shape[1],
        )


class RidgeModel:
    """Ridge regression with cross-validated alpha."""

    def __init__(self, alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
                 cv_folds: int = 5):
        self.name = "ridge"
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> PredictionResult:
        model = RidgeCV(alphas=self.alpha_grid, cv=self.cv_folds).fit(X_train, y_train)
        return PredictionResult(
            y_pred=model.predict(X_test),
            model_name=self.name,
            chosen_alpha=float(model.alpha_),
            n_features=X_train.shape[1],
        )


class LassoModel:
    """Lasso regression with cross-validated alpha."""

    def __init__(self, alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
                 cv_folds: int = 5):
        self.name = "lasso"
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> PredictionResult:
        model = LassoCV(
            alphas=self.alpha_grid, cv=self.cv_folds, max_iter=10000,
        ).fit(X_train, y_train)
        return PredictionResult(
            y_pred=model.predict(X_test),
            model_name=self.name,
            chosen_alpha=float(model.alpha_),
            n_features=X_train.shape[1],
        )


class ElasticNetModel:
    """ElasticNet with cross-validated alpha and l1_ratio."""

    def __init__(self, alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
                 l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99),
                 cv_folds: int = 5):
        self.name = "elasticnet"
        self.alpha_grid = alpha_grid
        self.l1_ratios = l1_ratios
        self.cv_folds = cv_folds

    def fit_predict(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> PredictionResult:
        model = ElasticNetCV(
            alphas=self.alpha_grid, l1_ratio=list(self.l1_ratios),
            cv=self.cv_folds, max_iter=10000,
        ).fit(X_train, y_train)
        return PredictionResult(
            y_pred=model.predict(X_test),
            model_name=self.name,
            chosen_alpha=float(model.alpha_),
            n_features=X_train.shape[1],
        )
