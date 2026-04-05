"""ExperimentRunner: orchestrates the full synthetic experiment."""

import os

# Prevent BLAS/OpenMP oversubscription when using multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.synthetic.config import AugmenterConfig, ExperimentConfig, ModelConfig, DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.evaluation.comparison import FairnessChecker, ResultTable
from src.synthetic.evaluation.metrics import ExperimentMetrics, compute_metrics

# Augmenter registry
from src.synthetic.augmenters.classical import (
    IdentityAugmenter,
    LogAbsAugmenter,
    OracleAugmenter,
    PolynomialAugmenter,
    RFFAugmenter,
)
from src.synthetic.augmenters.quantum_fixed import (
    AngleEncodingAugmenter,
    IQPAugmenter,
    ProbabilityAugmenter,
    QAOAAugmenter,
    ReservoirAugmenter,
    ZZMapAugmenter,
)
from src.synthetic.augmenters.quantum_learned import VQCAugmenter
from src.synthetic.augmenters.quantum_unified import UnifiedQuantumAugmenter

# Model registry
from src.synthetic.models.linear import (
    ElasticNetModel,
    LassoModel,
    OLSModel,
    RidgeModel,
)


def _build_augmenter(config: AugmenterConfig):
    """Instantiate an augmenter from its config."""
    registry = {
        "identity": lambda p: IdentityAugmenter(),
        "poly_deg2_interact": lambda p: PolynomialAugmenter(degree=2, interaction_only=True),
        "poly_deg2": lambda p: PolynomialAugmenter(degree=2),
        "poly_deg3": lambda p: PolynomialAugmenter(degree=3),
        "interaction_log": lambda p: LogAbsAugmenter(),
        "oracle": lambda p: OracleAugmenter(),
        "rff": lambda p: RFFAugmenter(**p),
        "angle_basic": lambda p: AngleEncodingAugmenter(use_strongly_entangling=False, **p),
        "angle_strong": lambda p: AngleEncodingAugmenter(use_strongly_entangling=True, **p),
        "zz": lambda p: ZZMapAugmenter(**p),
        "iqp": lambda p: IQPAugmenter(**p),
        "reservoir": lambda p: ReservoirAugmenter(**p),
        "qaoa": lambda p: QAOAAugmenter(**p),
        "prob": lambda p: ProbabilityAugmenter(**p),
        "vqc": lambda p: VQCAugmenter(**p),
        "unified": lambda p: UnifiedQuantumAugmenter(**p),
    }

    # Try neural augmenters (optional torch dependency)
    try:
        from src.synthetic.augmenters.neural import (
            AutoencoderAugmenter,
            LearnedRFFAugmenter,
            MLPAugmenter,
        )
        registry.update({
            "mlp": lambda p: MLPAugmenter(**p),
            "autoencoder": lambda p: AutoencoderAugmenter(**p),
            "learned_rff": lambda p: LearnedRFFAugmenter(**p),
        })
    except ImportError:
        pass

    # Match by prefix: "rff_10" -> key "rff", "angle_strong_4q_2L" -> key "angle_strong"
    for key in sorted(registry.keys(), key=len, reverse=True):
        if config.name.startswith(key):
            return registry[key](config.params)

    raise ValueError(f"Unknown augmenter: {config.name!r}")


def _build_model(config: ModelConfig):
    """Instantiate a model from its config."""
    models = {
        "ols": lambda: OLSModel(),
        "ridge": lambda: RidgeModel(alpha_grid=config.alpha_grid, cv_folds=config.cv_folds),
        "lasso": lambda: LassoModel(alpha_grid=config.alpha_grid, cv_folds=config.cv_folds),
        "elasticnet": lambda: ElasticNetModel(alpha_grid=config.alpha_grid, cv_folds=config.cv_folds),
    }
    if config.name not in models:
        raise ValueError(f"Unknown model: {config.name!r}")
    return models[config.name]()


def _run_single_augmenter(aug_config, X_train, X_test, y_train, clip_range):
    """Run a single augmenter (for multiprocessing). Returns (name, train_result, test_result)."""
    augmenter = _build_augmenter(aug_config)

    if aug_config.kind.startswith("quantum") and clip_range is not None:
        X_tr = np.clip(X_train, -clip_range, clip_range)
        X_te = np.clip(X_test, -clip_range, clip_range)
    else:
        X_tr, X_te = X_train, X_test

    augmenter.fit(X_tr, y_train)
    train_result = augmenter.transform(X_tr)
    test_result = augmenter.transform(X_te)
    return aug_config.name, train_result, test_result


class ExperimentRunner:
    """Orchestrates the full experiment: data -> augmenters -> models -> metrics."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fairness = FairnessChecker()
        run_dir = str(Path(config.results_dir) / config.run_id)
        self.result_table = ResultTable(run_dir)
        self._n_workers = os.cpu_count()

    def _result_exists(self, aug_name: str, model_name: str, seed: int) -> bool:
        """Check if a result JSON already exists for this (augmenter, model, seed)."""
        path = Path(self.result_table.results_dir) / f"seed_{seed}" / f"{aug_name}__{model_name}.json"
        return path.exists()

    def _needs_augmenter(self, aug_name: str, seed: int) -> bool:
        """Check if any model result is missing for this augmenter+seed."""
        if aug_name in self.config.force_rerun:
            return True
        return any(
            not self._result_exists(aug_name, m.name, seed)
            for m in self.config.models
        )

    def run(self) -> list[ExperimentMetrics]:
        all_metrics = []
        run_dir = Path(self.config.results_dir) / self.config.run_id
        print(f"Results directory: {run_dir}")
        print(f"Workers: {self._n_workers}")
        if self.config.force_rerun:
            print(f"Force rerun: {self.config.force_rerun}")

        seed_pbar = tqdm(self.config.seeds, desc="Seeds", unit="seed")
        for seed in seed_pbar:
            seed_pbar.set_postfix(seed=seed)

            # Filter to augmenters that need running for this seed
            needed = [c for c in self.config.augmenters if self._needs_augmenter(c.name, seed)]
            skipped = len(self.config.augmenters) - len(needed)
            if skipped:
                tqdm.write(f"  Seed {seed}: {skipped} augmenters cached, {len(needed)} to run")
            if not needed:
                continue

            dgp_config = DGPConfig(
                n_train=self.config.dgp.n_train,
                n_test=self.config.dgp.n_test,
                regime1_prob=self.config.dgp.regime1_prob,
                seed=seed,
            )

            # 1. Load or generate data (now includes regimes)
            X_train, X_test, y_train, y_test, regime_train, regime_test = get_or_generate(
                dgp_config, self.config.data_dir,
            )

            # 2. Scale ONCE
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # 3. Run needed augmenters (parallel for fixed, sequential for learning-based)
            aug_results_train = {}
            aug_results_test = {}

            fixed_configs = [c for c in needed if c.kind in ("classical", "quantum_fixed")]
            learned_configs = [c for c in needed if c.kind not in ("classical", "quantum_fixed")]

            # Run fixed augmenters in parallel
            if fixed_configs:
                with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
                    futures = {
                        pool.submit(
                            _run_single_augmenter, cfg, X_train_s, X_test_s, y_train, self.config.clip_range,
                        ): cfg.name
                        for cfg in fixed_configs
                    }
                    aug_pbar = tqdm(
                        as_completed(futures), total=len(futures),
                        desc=f"  Augmenters (seed={seed})", unit="aug", leave=False,
                    )
                    for future in aug_pbar:
                        name, train_res, test_res = future.result()
                        aug_results_train[name] = train_res
                        aug_results_test[name] = test_res
                        aug_pbar.set_postfix(last=name, feats=train_res.n_original + train_res.n_augmented)

            # Run learning-based augmenters sequentially
            for aug_config in learned_configs:
                name, train_res, test_res = _run_single_augmenter(
                    aug_config, X_train_s, X_test_s, y_train, self.config.clip_range,
                )
                aug_results_train[name] = train_res
                aug_results_test[name] = test_res

            # 4. Save features + dimensionality report
            if aug_results_train:
                feat_seed_dir = Path(self.config.features_dir) / self.config.run_id / f"seed_{seed}"
                feat_seed_dir.mkdir(parents=True, exist_ok=True)
                for aug_name in aug_results_train:
                    feat_path = feat_seed_dir / f"{aug_name}.npz"
                    np.savez_compressed(
                        feat_path,
                        train=aug_results_train[aug_name].features,
                        test=aug_results_test[aug_name].features,
                    )
                dim_report = self.fairness.report_dimensionality(aug_results_train)
                tqdm.write(f"\n  Seed {seed} — Dimensionality report:\n{dim_report.to_string(index=False)}")

            # 5. Run all models on all augmented feature sets
            for aug_name in aug_results_train:
                train_result = aug_results_train[aug_name]
                test_result = aug_results_test[aug_name]

                for model_config in self.config.models:
                    # Skip if result exists and not force-rerunning
                    if (aug_name not in self.config.force_rerun
                            and self._result_exists(aug_name, model_config.name, seed)):
                        continue

                    t0 = time.perf_counter()
                    model = _build_model(model_config)
                    pred = model.fit_predict(
                        train_result.features, y_train, test_result.features,
                    )
                    total_elapsed = time.perf_counter() - t0

                    metrics = compute_metrics(
                        y_test, pred, train_result, seed, total_elapsed,
                        augmenter_name_override=aug_name,
                        regime_test=regime_test,
                        y_train=y_train,
                        regime_train=regime_train,
                    )
                    self.result_table.save_single(metrics)
                    all_metrics.append(metrics)

        # 6. Aggregate summary (loads ALL results in run dir, including cached)
        summary = self.result_table.summarize()
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Summary (mean across seeds) — {run_dir}")
        tqdm.write(f"{'='*60}")
        if not summary.empty:
            tqdm.write(summary.to_string(index=False))

        return all_metrics
