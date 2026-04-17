"""Walk-forward backtesting runner for real financial data."""

import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from src.real.config import ExperimentConfig
from src.real.data import FEATURE_COLS, REGIME_COLS, build_dataset
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


def _precompute_single_quantum(aug_config, X_all_clipped, X_train_clipped, y_train):
    """Build and run a single quantum augmenter on the full dataset.

    Top-level function so it can be pickled for ProcessPoolExecutor.
    Each worker builds its own augmenter (avoids pickling QNode objects).
    """
    if aug_config.name.startswith("qunified"):
        from src.real.quantum_unified_real import UnifiedReservoirAugmenter
        aug = UnifiedReservoirAugmenter(**aug_config.params)
    elif aug_config.name.startswith("qres"):
        aug = QuantumReservoirAugmenter(**aug_config.params)
    else:
        raise ValueError(f"Unknown quantum augmenter: {aug_config.name}")

    aug.fit(X_train_clipped, y_train)
    result = aug.transform(X_all_clipped)

    meta = {
        "n_features_total": result.n_original + result.n_augmented,
        "n_features_augmented": result.n_augmented,
        "n_trainable_params": result.n_trainable_params,
        "n_random_params": result.n_random_params,
        "circuit_depth": result.circuit_depth,
        "qubit_count": result.qubit_count,
        "gate_count": result.gate_count,
    }
    return result.features, meta


def _run_single_augmenter_real(aug_config, X_train, X_test, y_train, clip_range):
    """Run a single augmenter, with support for quantum reservoir configs."""
    if aug_config.name.startswith("qres"):
        augmenter = QuantumReservoirAugmenter(**aug_config.params)
        if clip_range is not None:
            X_train = np.clip(X_train, -clip_range, clip_range)
            X_test = np.clip(X_test, -clip_range, clip_range)
        augmenter.fit(X_train, y_train)
        train_result = augmenter.transform(X_train)
        test_result = augmenter.transform(X_test)
        return aug_config.name, train_result, test_result
    elif aug_config.name.startswith("qunified"):
        from src.real.quantum_unified_real import UnifiedReservoirAugmenter
        augmenter = UnifiedReservoirAugmenter(**aug_config.params)
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
    """Walk-forward backtesting with monthly scaler refit and cached augmentation.

    Optimizations over naive per-window augmentation:
    1. Augmenter objects built once and reused across all windows
    2. Scaler refit + full-dataset augmentation every SCALER_REFIT_INTERVAL trading days
       (consecutive windows share >99% of training data, so the scaler barely moves)
    3. Per-window work reduced to array slicing + Ridge/Lasso fit (~ms)
    """

    SCALER_REFIT_INTERVAL = 21  # trading days (~monthly)

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _get_eval_dates(self, dataset: pd.DataFrame) -> list[pd.Timestamp]:
        """Get evaluation dates: every step_days trading days after warmup.

        Uses trading-day indexing so train_window_days and gap_days refer to
        actual trading days, not calendar days.
        """
        all_dates = sorted(dataset["date"].unique())
        self._trading_dates = all_dates  # cache for _get_masks
        bc = self.config.backtest

        # First eval date needs train_window_days + gap_days trading days before it
        first_eval_idx = bc.train_window_days + bc.gap_days
        if first_eval_idx >= len(all_dates):
            return []

        return all_dates[first_eval_idx::bc.step_days]

    def _get_masks(self, dataset: pd.DataFrame, eval_date: pd.Timestamp):
        """Get train and test masks for a given evaluation date.

        Uses trading-day indexing: train window is the preceding
        train_window_days trading days, with a gap_days trading-day gap
        before the eval date.
        """
        bc = self.config.backtest
        dates = self._trading_dates
        eval_idx = dates.index(eval_date)

        # Train: trading days [eval_idx - gap - window, eval_idx - gap)
        train_end_idx = eval_idx - bc.gap_days
        train_start_idx = max(0, train_end_idx - bc.train_window_days)

        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]

        train_mask = (dataset["date"] >= train_start) & (dataset["date"] <= train_end)
        test_mask = dataset["date"] == eval_date

        return train_mask, test_mask

    @staticmethod
    def _build_augmenter_for_config(aug_config):
        """Build a single augmenter object from config. Returns (augmenter, is_quantum)."""
        if aug_config.name.startswith("qunified"):
            from src.real.quantum_unified_real import UnifiedReservoirAugmenter
            return UnifiedReservoirAugmenter(**aug_config.params), True
        elif aug_config.name.startswith("qres"):
            return QuantumReservoirAugmenter(**aug_config.params), True
        else:
            return _build_augmenter(aug_config), False

    def _precompute_features(self, X_all, y_train, train_mask, augmenters, aug_is_quantum,
                             corr_all=None, date_row_idx=None):
        """Refit scaler on training window, augment full dataset, cache results.

        Quantum augmenters run in parallel via ProcessPoolExecutor (each worker
        builds its own augmenter to avoid pickling QNode objects). Classical
        augmenters run in the main process (fast, no benefit from parallelism).

        If corr_all is provided and config.corr_augmenter is set, also runs a
        dedicated quantum augmenter on the correlation matrix features and appends
        its output to every augmenter's cached features.

        Returns (cached_features, aug_metadata) where cached_features maps
        augmenter name -> full-dataset augmented feature array.
        """
        clip_range = self.config.clip_range

        # Fit scaler on training window only
        scaler = StandardScaler().fit(X_all[train_mask])
        X_scaled = scaler.transform(X_all)
        X_clipped = np.clip(X_scaled, -clip_range, clip_range) if clip_range else X_scaled

        cached = {}
        metadata = {}

        # Classical augmenters: fast, run in main process
        for name, aug in augmenters.items():
            if aug_is_quantum[name]:
                continue
            aug.fit(X_scaled[train_mask], y_train)
            result = aug.transform(X_scaled)
            cached[name] = result.features
            metadata[name] = self._extract_metadata(result)

        # Quantum augmenters: run in parallel worker processes
        quantum_configs = [
            cfg for cfg in self.config.augmenters if aug_is_quantum.get(cfg.name)
        ]
        if quantum_configs:
            X_train_q = X_clipped[train_mask]
            y_train_q = y_train
            n_workers = min(len(quantum_configs), os.cpu_count() or 1)

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _precompute_single_quantum, cfg, X_clipped, X_train_q, y_train_q,
                    ): cfg.name
                    for cfg in quantum_configs
                }
                for future in as_completed(futures):
                    name = futures[future]
                    features, meta = future.result()
                    cached[name] = features
                    metadata[name] = meta

        # Correlation quantum channel (Approach 3)
        if self.config.corr_augmenter is not None and corr_all is not None:
            from src.real.quantum_unified_real import UnifiedReservoirAugmenter

            # Derive date-level train mask from row-level train mask
            train_row_indices = set(np.where(train_mask)[0])
            train_date_indices = set(date_row_idx[i] for i in train_row_indices)
            train_mask_dates = np.array([i in train_date_indices for i in range(len(corr_all))])

            # Scale correlation data separately (bounded [-1,1], different scale)
            corr_scaler = StandardScaler().fit(corr_all[train_mask_dates])
            corr_scaled = corr_scaler.transform(corr_all)
            if clip_range:
                corr_scaled = np.clip(corr_scaled, -clip_range, clip_range)

            # Build and run dedicated correlation augmenter
            corr_aug = UnifiedReservoirAugmenter(**self.config.corr_augmenter)
            dummy_y = np.zeros(np.count_nonzero(train_mask_dates))
            corr_aug.fit(corr_scaled[train_mask_dates], dummy_y)
            corr_result = corr_aug.transform(corr_scaled)

            # Extract only quantum features (not the 45 raw correlation inputs)
            corr_quantum = corr_result.features[:, corr_result.n_original:]

            # Broadcast from date-level (n_dates, q) to row-level (n_rows, q)
            corr_broadcast = corr_quantum[date_row_idx]

            # Append to ALL cached augmenter features and update metadata
            n_corr_features = corr_broadcast.shape[1]
            for name in cached:
                cached[name] = np.hstack([cached[name], corr_broadcast])
                metadata[name]["n_features_total"] += n_corr_features
                metadata[name]["n_corr_quantum_features"] = n_corr_features

        return cached, metadata

    @staticmethod
    def _extract_metadata(result):
        return {
            "n_features_total": result.n_original + result.n_augmented,
            "n_features_augmented": result.n_augmented,
            "n_trainable_params": result.n_trainable_params,
            "n_random_params": result.n_random_params,
            "circuit_depth": result.circuit_depth,
            "qubit_count": result.qubit_count,
            "gate_count": result.gate_count,
        }

    def run(self) -> dict:
        """Run the full walk-forward backtest.

        Returns a dict with keys:
          - "metrics": dict of (aug_name, model_name) -> {mse, mae, pearson_r, ...}
          - "predictions": dict of (aug_name, model_name) -> DataFrame of OOS predictions
        """
        print("Loading dataset...")
        dataset, corr_triangle_df = build_dataset(self.config.data)
        print(f"Dataset: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers, "
              f"dates {dataset['date'].min().date()} to {dataset['date'].max().date()}")

        eval_dates = self._get_eval_dates(dataset)
        print(f"Evaluation dates: {len(eval_dates)} "
              f"({eval_dates[0].date()} to {eval_dates[-1].date()})")

        # Build all augmenters once (persistent across windows)
        augmenters = {}
        aug_is_quantum = {}
        for cfg in self.config.augmenters:
            aug, is_q = self._build_augmenter_for_config(cfg)
            augmenters[cfg.name] = aug
            aug_is_quantum[cfg.name] = is_q
        print(f"Augmenters: {len(augmenters)} ({sum(aug_is_quantum.values())} quantum)"
              + (f" + corr quantum" if self.config.corr_augmenter else ""))

        # Per-stock features: 14 original, optionally + 3 regime
        feature_cols = FEATURE_COLS + REGIME_COLS if self.config.use_regime_features else FEATURE_COLS
        X_all = dataset[feature_cols].values

        # Prepare correlation data for quantum encoding (Approach 3)
        corr_all = None
        date_row_idx = None
        if self.config.corr_augmenter is not None:
            unique_dates = sorted(dataset["date"].unique())
            date_to_idx = {d: i for i, d in enumerate(unique_dates)}
            corr_all = corr_triangle_df.reindex(unique_dates).values
            date_row_idx = np.array([date_to_idx[d] for d in dataset["date"]])
        y_all = dataset["target"].values

        # Collect all OOS predictions per (augmenter, model)
        oos_predictions: dict[tuple[str, str], list] = defaultdict(list)
        aug_metadata: dict[str, dict] = {}

        # Cached augmented features (recomputed on scaler refit)
        cached_features = None
        refit_countdown = 0

        for idx, t in enumerate(tqdm(eval_dates, desc="Backtest", unit="day")):
            train_mask, test_mask = self._get_masks(dataset, t)

            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            test_tickers = dataset.loc[test_mask, "ticker"].values

            if y_train.sum() == 0 or len(y_test) == 0:
                continue
            if np.count_nonzero(train_mask) < self.config.backtest.min_train_samples:
                continue

            # Refit scaler and re-augment monthly
            if refit_countdown <= 0:
                t0 = time.perf_counter()
                cached_features, new_meta = self._precompute_features(
                    X_all, y_train, train_mask, augmenters, aug_is_quantum,
                    corr_all=corr_all, date_row_idx=date_row_idx,
                )
                aug_metadata.update(new_meta)
                elapsed = time.perf_counter() - t0
                if idx == 0:
                    print(f"Precompute: {elapsed:.1f}s for {len(X_all)} rows × "
                          f"{len(augmenters)} augmenters")
                refit_countdown = self.SCALER_REFIT_INTERVAL

            refit_countdown -= 1

            # Slice cached features and fit models (fast: ~ms per model)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            for aug_name in augmenters:
                X_train_aug = cached_features[aug_name][train_idx]
                X_test_aug = cached_features[aug_name][test_idx]

                for model_cfg in self.config.models:
                    model = _build_model(model_cfg)
                    pred = model.fit_predict(X_train_aug, y_train, X_test_aug)

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
