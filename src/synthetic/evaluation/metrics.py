"""Evaluation metrics and ExperimentMetrics dataclass."""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.synthetic.augmenters.base import AugmenterResult
from src.synthetic.evaluation.complexity import compute_complexity_metrics
from src.synthetic.models.base import PredictionResult


def _safe_pearsonr(y_true, y_pred):
    """Pearson r that handles constant arrays gracefully."""
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


def _compute_suite(y_true, y_pred, regimes=None):
    """Compute MSE/MAE/corr overall and per-regime. Returns dict of metrics."""
    m = {
        "mse": round(float(mean_squared_error(y_true, y_pred)), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "pearson_r": round(_safe_pearsonr(y_true, y_pred), 6),
    }
    if regimes is not None:
        for rid in (1, 2):
            mask = regimes == rid
            if mask.sum() > 0:
                m[f"mse_regime{rid}"] = round(float(mean_squared_error(y_true[mask], y_pred[mask])), 6)
                m[f"mae_regime{rid}"] = round(float(mean_absolute_error(y_true[mask], y_pred[mask])), 6)
                m[f"pearson_r_regime{rid}"] = round(_safe_pearsonr(y_true[mask], y_pred[mask]), 6)
    return m


@dataclass
class ExperimentMetrics:
    """Full metrics for one (seed, augmenter, model) run."""
    # Test metrics (aggregate)
    mse: float
    mae: float
    pearson_r: float
    # Test per-regime
    mse_regime1: float | None = None
    mae_regime1: float | None = None
    pearson_r_regime1: float | None = None
    mse_regime2: float | None = None
    mae_regime2: float | None = None
    pearson_r_regime2: float | None = None
    # Train metrics (aggregate)
    mse_train: float | None = None
    mae_train: float | None = None
    pearson_r_train: float | None = None
    # Train per-regime
    mse_train_regime1: float | None = None
    mse_train_regime2: float | None = None
    pearson_r_train_regime1: float | None = None
    pearson_r_train_regime2: float | None = None
    # Augmenter info
    augmenter_name: str = ""
    n_features_total: int = 0
    n_features_augmented: int = 0
    augmenter_wall_clock: float = 0.0
    # Parameter counts
    n_trainable_params: int = 0
    n_random_params: int = 0
    # Complexity metrics
    effective_rank: float | None = None
    nonlinearity_score: float | None = None
    feature_target_alignment: float | None = None
    coef_l2_norm: float | None = None
    lasso_active_fraction: float | None = None
    # Quantum resource info
    circuit_depth: int | None = None
    qubit_count: int | None = None
    gate_count: int | None = None
    # Model info
    model_name: str = ""
    chosen_alpha: float | None = None
    # Metadata
    seed: int = 0
    total_wall_clock: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_test: np.ndarray,
    pred: PredictionResult,
    aug: AugmenterResult,
    seed: int,
    total_elapsed: float,
    augmenter_name_override: str | None = None,
    regime_test: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    regime_train: np.ndarray | None = None,
) -> ExperimentMetrics:
    """Compute all metrics for one experiment run, including per-regime and train."""
    # Test metrics
    test = _compute_suite(y_test, pred.y_pred, regime_test)

    # Train metrics
    train = {}
    if y_train is not None and pred.y_train_pred is not None:
        t = _compute_suite(y_train, pred.y_train_pred, regime_train)
        train = {
            "mse_train": t["mse"],
            "mae_train": t["mae"],
            "pearson_r_train": t["pearson_r"],
            "mse_train_regime1": t.get("mse_regime1"),
            "mse_train_regime2": t.get("mse_regime2"),
            "pearson_r_train_regime1": t.get("pearson_r_regime1"),
            "pearson_r_train_regime2": t.get("pearson_r_regime2"),
        }

    # Complexity metrics from feature matrix
    cx = compute_complexity_metrics(aug.features, y_train if y_train is not None else y_test, aug.n_original)

    return ExperimentMetrics(
        mse=test["mse"],
        mae=test["mae"],
        pearson_r=test["pearson_r"],
        mse_regime1=test.get("mse_regime1"),
        mae_regime1=test.get("mae_regime1"),
        pearson_r_regime1=test.get("pearson_r_regime1"),
        mse_regime2=test.get("mse_regime2"),
        mae_regime2=test.get("mae_regime2"),
        pearson_r_regime2=test.get("pearson_r_regime2"),
        **train,
        augmenter_name=augmenter_name_override or aug.augmenter_name,
        n_features_total=aug.n_original + aug.n_augmented,
        n_features_augmented=aug.n_augmented,
        augmenter_wall_clock=round(aug.wall_clock_seconds, 4),
        n_trainable_params=aug.n_trainable_params,
        n_random_params=aug.n_random_params,
        effective_rank=cx.get("effective_rank"),
        nonlinearity_score=cx.get("nonlinearity_score"),
        feature_target_alignment=cx.get("feature_target_alignment"),
        coef_l2_norm=pred.coef_l2_norm,
        lasso_active_fraction=pred.lasso_active_fraction,
        circuit_depth=aug.circuit_depth,
        qubit_count=aug.qubit_count,
        gate_count=aug.gate_count,
        model_name=pred.model_name,
        chosen_alpha=pred.chosen_alpha,
        seed=seed,
        total_wall_clock=round(total_elapsed, 4),
    )
