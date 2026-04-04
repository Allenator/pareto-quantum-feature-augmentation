"""ExperimentRunner: orchestrates the full synthetic experiment."""

import time

import numpy as np
from sklearn.preprocessing import StandardScaler

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

    # Match by prefix: "rff_10" -> key "rff", "angle_strong_2L" -> key "angle_strong"
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


class ExperimentRunner:
    """Orchestrates the full experiment: data → augmenters → models → metrics."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fairness = FairnessChecker()
        self.result_table = ResultTable(config.results_dir)

    def run(self) -> list[ExperimentMetrics]:
        all_metrics = []

        for seed in self.config.seeds:
            print(f"\n{'='*60}")
            print(f"Seed {seed}")
            print(f"{'='*60}")

            dgp_config = DGPConfig(
                n_train=self.config.dgp.n_train,
                n_test=self.config.dgp.n_test,
                regime1_prob=self.config.dgp.regime1_prob,
                seed=seed,
            )

            # 1. Load or generate data
            X_train, X_test, y_train, y_test = get_or_generate(
                dgp_config, self.config.data_dir,
            )

            # 2. Scale ONCE
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Clip for quantum circuits if configured
            if self.config.clip_range is not None:
                X_train_clipped = np.clip(X_train_s, -self.config.clip_range, self.config.clip_range)
                X_test_clipped = np.clip(X_test_s, -self.config.clip_range, self.config.clip_range)
            else:
                X_train_clipped = X_train_s
                X_test_clipped = X_test_s

            # 3. Run all augmenters
            aug_results_train = {}
            aug_results_test = {}

            for aug_config in self.config.augmenters:
                print(f"\n  Augmenter: {aug_config.name} ({aug_config.kind})")
                augmenter = _build_augmenter(aug_config)

                # Use clipped data for quantum augmenters
                if aug_config.kind.startswith("quantum"):
                    X_tr, X_te = X_train_clipped, X_test_clipped
                else:
                    X_tr, X_te = X_train_s, X_test_s

                augmenter.fit(X_tr, y_train)
                aug_results_train[aug_config.name] = augmenter.transform(X_tr)
                aug_results_test[aug_config.name] = augmenter.transform(X_te)

                r = aug_results_train[aug_config.name]
                print(f"    Features: {r.n_original} orig + {r.n_augmented} new = {r.n_original + r.n_augmented} total")
                print(f"    Time: {r.wall_clock_seconds:.3f}s")
                if r.circuit_depth is not None:
                    print(f"    Circuit: depth={r.circuit_depth}, qubits={r.qubit_count}")

            # 4. Fairness check (use unclipped data for classical, clipped for quantum)
            # Skip strict original-feature check since quantum augmenters may use clipped inputs
            dim_report = self.fairness.report_dimensionality(aug_results_train)
            print(f"\n  Dimensionality report:\n{dim_report.to_string(index=False)}")

            # 5. Run all models on all augmented feature sets
            for aug_name in aug_results_train:
                train_result = aug_results_train[aug_name]
                test_result = aug_results_test[aug_name]

                for model_config in self.config.models:
                    t0 = time.perf_counter()
                    model = _build_model(model_config)
                    pred = model.fit_predict(
                        train_result.features, y_train, test_result.features,
                    )
                    total_elapsed = time.perf_counter() - t0

                    metrics = compute_metrics(
                        y_test, pred, train_result, seed, total_elapsed,
                        augmenter_name_override=aug_name,
                    )
                    self.result_table.save_single(metrics)
                    all_metrics.append(metrics)

                    print(f"    {aug_name} + {model_config.name}: "
                          f"MSE={metrics.mse:.4f} MAE={metrics.mae:.4f} "
                          f"r={metrics.pearson_r:.4f} (α={pred.chosen_alpha})")

        # 6. Aggregate summary
        summary = self.result_table.summarize()
        print(f"\n{'='*60}")
        print("Summary (mean across seeds):")
        print(f"{'='*60}")
        if not summary.empty:
            print(summary.to_string(index=False))

        return all_metrics
