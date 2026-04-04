"""Evaluation metrics and ExperimentMetrics dataclass."""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.synthetic.augmenters.base import AugmenterResult
from src.synthetic.models.base import PredictionResult


@dataclass
class ExperimentMetrics:
    """Full metrics for one (seed, augmenter, model) run."""
    # Core metrics
    mse: float
    mae: float
    pearson_r: float
    # Augmenter info
    augmenter_name: str
    n_features_total: int
    n_features_augmented: int
    augmenter_wall_clock: float
    # Quantum resource info
    circuit_depth: int | None
    qubit_count: int | None
    gate_count: int | None
    # Model info
    model_name: str
    chosen_alpha: float | None
    # Metadata
    seed: int
    total_wall_clock: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    pred: PredictionResult,
    aug: AugmenterResult,
    seed: int,
    total_elapsed: float,
    augmenter_name_override: str | None = None,
) -> ExperimentMetrics:
    """Compute all metrics for one experiment run."""
    mse = float(mean_squared_error(y_true, pred.y_pred))
    mae = float(mean_absolute_error(y_true, pred.y_pred))
    corr = float(pearsonr(y_true, pred.y_pred)[0])

    return ExperimentMetrics(
        mse=round(mse, 6),
        mae=round(mae, 6),
        pearson_r=round(corr, 6),
        augmenter_name=augmenter_name_override or aug.augmenter_name,
        n_features_total=aug.n_original + aug.n_augmented,
        n_features_augmented=aug.n_augmented,
        augmenter_wall_clock=round(aug.wall_clock_seconds, 4),
        circuit_depth=aug.circuit_depth,
        qubit_count=aug.qubit_count,
        gate_count=aug.gate_count,
        model_name=pred.model_name,
        chosen_alpha=pred.chosen_alpha,
        seed=seed,
        total_wall_clock=round(total_elapsed, 4),
    )
