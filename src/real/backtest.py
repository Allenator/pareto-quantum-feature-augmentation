"""Walk-forward backtesting runner for real financial data."""

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from src.real.config import ExperimentConfig
from src.real.data import FEATURE_COLS, build_dataset
from src.synthetic.augmenters.base import AugmenterResult, _make_result
from src.synthetic.runner import _build_augmenter, _build_model, _run_single_augmenter

# Import the standalone QuantumReservoir (with final Rot layer)
from scripts.quantum_reservoir import QuantumReservoir


def _safe_pearsonr(y_true, y_pred):
    """Pearson r that handles constant arrays gracefully."""
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(pearsonr(y_true, y_pred)[0])


class QuantumReservoirAugmenter:
    """Wrapper around scripts/quantum_reservoir.py QuantumReservoir.

    Adapts it to the FeatureAugmenter protocol (fit/transform -> AugmenterResult).
    This version has a final Rot layer after the last CNOT, giving more expressivity.
    """

    def __init__(self, n_qubits: int = 14, n_layers: int = 2,
                 n_ensemble: int = 3, seed: int = 42):
        self._qr = QuantumReservoir(
            n_qubits=n_qubits, n_layers=n_layers,
            n_ensemble=n_ensemble, seed=seed,
        )
        self.name = f"qres_{n_ensemble}x{n_layers}_{n_qubits}q"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass  # Fixed random parameters

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        X_new = self._qr.transform(X)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=self._qr.n_random_params,
            qubit_count=self._qr.n_qubits,
        )


def _run_single_augmenter_real(aug_config, X_train, X_test, y_train, clip_range):
    """Run a single augmenter, with support for QuantumReservoir configs."""
    if aug_config.name.startswith("qres"):
        augmenter = QuantumReservoirAugmenter(**aug_config.params)
        if clip_range is not None:
            X_train = np.clip(X_train, -clip_range, clip_range)
            X_test = np.clip(X_test, -clip_range, clip_range)
        augmenter.fit(X_train, y_train)
        train_result = augmenter.transform(X_train)
        test_result = augmenter.transform(X_test)
        return aug_config.name, train_result, test_result
    else:
        return _run_single_augmenter(aug_config, X_train, X_test, y_train, clip_range)


class BacktestRunner:
    """Walk-forward backtesting: rolling 2-year train, predict next day, roll forward."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._n_workers = os.cpu_count()

    def _get_eval_dates(self, dataset: pd.DataFrame) -> list[pd.Timestamp]:
        """Get evaluation dates: every step_days trading days after warmup."""
        all_dates = sorted(dataset["date"].unique())
        bc = self.config.backtest

        # Warmup: need at least train_window_days of calendar days
        first_date = all_dates[0]
        warmup_end = first_date + pd.Timedelta(days=bc.train_window_days)

        valid_dates = [d for d in all_dates if d >= warmup_end]
        return valid_dates[::bc.step_days]

    def _get_masks(self, dataset: pd.DataFrame, eval_date: pd.Timestamp):
        """Get train and test masks for a given evaluation date."""
        bc = self.config.backtest

        # Train: [eval_date - 2yr, eval_date - gap)
        train_end = eval_date - pd.Timedelta(days=bc.gap_days)
        train_start = eval_date - pd.Timedelta(days=bc.train_window_days)

        train_mask = (dataset["date"] >= train_start) & (dataset["date"] <= train_end)
        test_mask = dataset["date"] == eval_date

        return train_mask, test_mask

    def _run_augmenters_for_window(self, X_train_s, X_test_s, y_train):
        """Run all augmenters for one backtest window. Returns dict of (train_result, test_result)."""
        aug_results = {}

        fixed_configs = [c for c in self.config.augmenters if c.kind in ("classical", "quantum_fixed")]
        learned_configs = [c for c in self.config.augmenters if c.kind not in ("classical", "quantum_fixed")]

        # Fixed augmenters in parallel
        if fixed_configs:
            with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                futures = {
                    pool.submit(
                        _run_single_augmenter_real, cfg, X_train_s, X_test_s, y_train, self.config.clip_range,
                    ): cfg.name
                    for cfg in fixed_configs
                }
                for future in as_completed(futures):
                    name, train_res, test_res = future.result()
                    aug_results[name] = (train_res, test_res)

        # Learning-based augmenters sequentially
        for cfg in learned_configs:
            name, train_res, test_res = _run_single_augmenter(
                cfg, X_train_s, X_test_s, y_train, self.config.clip_range,
            )
            aug_results[name] = (train_res, test_res)

        return aug_results

    def run(self) -> dict:
        """Run the full walk-forward backtest.

        Returns a dict with keys:
          - "metrics": dict of (aug_name, model_name) -> {mse, mae, pearson_r, ...}
          - "predictions": dict of (aug_name, model_name) -> DataFrame of OOS predictions
        """
        print(f"Loading dataset...")
        dataset = build_dataset(self.config.data)
        print(f"Dataset: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers, "
              f"dates {dataset['date'].min().date()} to {dataset['date'].max().date()}")

        eval_dates = self._get_eval_dates(dataset)
        print(f"Evaluation dates: {len(eval_dates)} "
              f"({eval_dates[0].date()} to {eval_dates[-1].date()})")
        print(f"Workers: {self._n_workers}")

        # Collect all OOS predictions per (augmenter, model)
        oos_predictions: dict[tuple[str, str], list] = defaultdict(list)

        # Track augmenter metadata from first window for final metrics
        aug_metadata: dict[str, dict] = {}

        for t in tqdm(eval_dates, desc="Backtest", unit="day"):
            train_mask, test_mask = self._get_masks(dataset, t)

            X_train = dataset.loc[train_mask, FEATURE_COLS].values
            y_train = dataset.loc[train_mask, "target"].values
            X_test = dataset.loc[test_mask, FEATURE_COLS].values
            y_test = dataset.loc[test_mask, "target"].values
            test_tickers = dataset.loc[test_mask, "ticker"].values

            if len(X_train) < self.config.backtest.min_train_samples or len(X_test) == 0:
                continue

            # Scale (fit on train only)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Run augmenters
            aug_results = self._run_augmenters_for_window(X_train_s, X_test_s, y_train)

            # Run models and collect predictions
            for aug_name, (train_res, test_res) in aug_results.items():
                # Capture metadata from first window
                if aug_name not in aug_metadata:
                    aug_metadata[aug_name] = {
                        "n_features_total": train_res.n_original + train_res.n_augmented,
                        "n_features_augmented": train_res.n_augmented,
                        "n_trainable_params": train_res.n_trainable_params,
                        "n_random_params": train_res.n_random_params,
                        "circuit_depth": train_res.circuit_depth,
                        "qubit_count": train_res.qubit_count,
                        "gate_count": train_res.gate_count,
                    }

                for model_cfg in self.config.models:
                    model = _build_model(model_cfg)
                    pred = model.fit_predict(train_res.features, y_train, test_res.features)

                    for i in range(len(y_test)):
                        oos_predictions[(aug_name, model_cfg.name)].append({
                            "date": t,
                            "ticker": test_tickers[i],
                            "y_true": float(y_test[i]),
                            "y_pred": float(pred.y_pred[i]),
                        })

        # Aggregate results
        return self._aggregate_results(oos_predictions, aug_metadata)

    def _aggregate_results(self, oos_predictions, aug_metadata) -> dict:
        """Compute final metrics from collected OOS predictions and save results."""
        results_dir = Path(self.config.results_dir) / self.config.run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        all_metrics = {}
        all_pred_dfs = {}

        for (aug_name, model_name), preds in oos_predictions.items():
            df = pd.DataFrame(preds)
            df["date"] = pd.to_datetime(df["date"])
            all_pred_dfs[(aug_name, model_name)] = df

            y_true = df["y_true"].values
            y_pred = df["y_pred"].values

            meta = aug_metadata.get(aug_name, {})

            metrics = {
                "augmenter_name": aug_name,
                "model_name": model_name,
                "mse": round(float(mean_squared_error(y_true, y_pred)), 6),
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
                "pearson_r": round(_safe_pearsonr(y_true, y_pred), 6),
                "n_predictions": len(df),
                "n_eval_dates": df["date"].nunique(),
                "n_tickers": df["ticker"].nunique(),
                **meta,
            }

            # Rolling 63-day (3-month) IC for stability
            df_sorted = df.sort_values("date")
            rolling_ic = []
            dates = sorted(df_sorted["date"].unique())
            window_size = 63
            for i in range(window_size, len(dates)):
                window_dates = dates[i - window_size:i]
                window_df = df_sorted[df_sorted["date"].isin(window_dates)]
                if len(window_df) >= 10:
                    ic = _safe_pearsonr(window_df["y_true"].values, window_df["y_pred"].values)
                    rolling_ic.append(ic)

            if rolling_ic:
                metrics["ic_mean_rolling_3m"] = round(float(np.mean(rolling_ic)), 6)
                metrics["ic_std_rolling_3m"] = round(float(np.std(rolling_ic)), 6)

            all_metrics[(aug_name, model_name)] = metrics

            # Save per-(augmenter, model) JSON
            json_path = results_dir / f"{aug_name}__{model_name}.json"
            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Save predictions parquet for post-hoc analysis
        for (aug_name, model_name), df in all_pred_dfs.items():
            pred_path = results_dir / f"{aug_name}__{model_name}_predictions.parquet"
            df.to_parquet(pred_path, index=False)

        # Summary CSV
        if all_metrics:
            summary = pd.DataFrame(all_metrics.values())
            summary = summary.sort_values("mse")
            summary_path = results_dir / "summary.csv"
            summary.to_csv(summary_path, index=False)

            print(f"\n{'='*70}")
            print(f"Results — {results_dir}")
            print(f"{'='*70}")
            print(summary.to_string(index=False))

        return {"metrics": all_metrics, "predictions": all_pred_dfs}
