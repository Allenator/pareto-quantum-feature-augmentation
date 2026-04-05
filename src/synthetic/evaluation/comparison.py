"""Fairness checking and result aggregation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.synthetic.augmenters.base import AugmenterResult
from src.synthetic.evaluation.metrics import ExperimentMetrics


class FairnessChecker:
    """Validates that all augmenters receive identical data and scaling."""

    def validate_split(
        self, X_train: np.ndarray, X_test: np.ndarray,
        aug_results_train: dict[str, AugmenterResult],
        aug_results_test: dict[str, AugmenterResult],
    ) -> None:
        """Verify all augmenters' results contain the original features."""
        n_train, n_feat = X_train.shape
        for name, result in aug_results_train.items():
            assert result.features.shape[0] == n_train, (
                f"Augmenter {name}: expected {n_train} train samples, got {result.features.shape[0]}"
            )
            # First n_feat columns should be the original features
            np.testing.assert_array_equal(
                result.features[:, :n_feat], X_train,
                err_msg=f"Augmenter {name}: original features not preserved in train set",
            )
        n_test = X_test.shape[0]
        for name, result in aug_results_test.items():
            assert result.features.shape[0] == n_test, (
                f"Augmenter {name}: expected {n_test} test samples, got {result.features.shape[0]}"
            )
            np.testing.assert_array_equal(
                result.features[:, :n_feat], X_test,
                err_msg=f"Augmenter {name}: original features not preserved in test set",
            )

    def report_dimensionality(
        self, aug_results: dict[str, AugmenterResult],
    ) -> pd.DataFrame:
        """Report feature dimensions for all augmenters."""
        rows = []
        for name, result in aug_results.items():
            rows.append({
                "augmenter": name,
                "n_original": result.n_original,
                "n_augmented": result.n_augmented,
                "n_total": result.n_original + result.n_augmented,
            })
        return pd.DataFrame(rows)


class ResultTable:
    """Aggregates ExperimentMetrics across seeds into summary tables."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

    def save_single(self, metrics: ExperimentMetrics) -> Path:
        """Save one result as JSON."""
        seed_dir = self.results_dir / f"seed_{metrics.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{metrics.augmenter_name}__{metrics.model_name}.json"
        path = seed_dir / filename
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        return path

    def load_all(self) -> list[ExperimentMetrics]:
        """Load all saved results."""
        results = []
        for json_path in self.results_dir.rglob("*.json"):
            with open(json_path) as f:
                data = json.load(f)
            results.append(ExperimentMetrics(**data))
        return results

    def summarize(self) -> pd.DataFrame:
        """Aggregate results across seeds into mean ± std."""
        results = self.load_all()
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([r.to_dict() for r in results])
        group_cols = ["augmenter_name", "model_name"]
        agg_spec = {
            "mse_mean": ("mse", "mean"),
            "mse_std": ("mse", "std"),
            "mae_mean": ("mae", "mean"),
            "mae_std": ("mae", "std"),
            "corr_mean": ("pearson_r", "mean"),
            "corr_std": ("pearson_r", "std"),
            "mse_r1_mean": ("mse_regime1", "mean"),
            "mse_r1_std": ("mse_regime1", "std"),
            "mse_r2_mean": ("mse_regime2", "mean"),
            "mse_r2_std": ("mse_regime2", "std"),
            "corr_r1_mean": ("pearson_r_regime1", "mean"),
            "corr_r2_mean": ("pearson_r_regime2", "mean"),
        }
        # Training metrics (may be missing in old results)
        if "mse_train" in df.columns and df["mse_train"].notna().any():
            agg_spec.update({
                "mse_train_mean": ("mse_train", "mean"),
                "mse_train_std": ("mse_train", "std"),
                "mse_train_r1_mean": ("mse_train_regime1", "mean"),
                "mse_train_r2_mean": ("mse_train_regime2", "mean"),
                "corr_train_mean": ("pearson_r_train", "mean"),
            })
        # Complexity metrics (may be missing in old results)
        for col, agg_fn in [
            ("n_trainable_params", "first"),
            ("n_random_params", "first"),
            ("effective_rank", "mean"),
            ("nonlinearity_score", "mean"),
            ("feature_target_alignment", "mean"),
            ("coef_l2_norm", "mean"),
            ("lasso_active_fraction", "mean"),
        ]:
            if col in df.columns and df[col].notna().any():
                agg_spec[col] = (col, agg_fn)
        agg_spec.update({
            "n_features": ("n_features_total", "first"),
            "n_augmented": ("n_features_augmented", "first"),
            "circuit_depth": ("circuit_depth", "first"),
            "qubit_count": ("qubit_count", "first"),
            "wall_clock_mean": ("total_wall_clock", "mean"),
            "n_seeds": ("seed", "count"),
        })
        agg = df.groupby(group_cols).agg(**agg_spec).reset_index()

        summary_path = self.results_dir / "summary.csv"
        agg.to_csv(summary_path, index=False)
        return agg
