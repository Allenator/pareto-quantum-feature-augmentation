"""Frozen dataclasses for all experiment configuration."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DGPConfig:
    """Data generating process configuration."""
    n_train: int = 10_000
    n_test: int = 10_000
    regime1_prob: float = 0.75
    seed: int = 42


@dataclass(frozen=True)
class AugmenterConfig:
    """Feature augmenter configuration."""
    name: str           # e.g. "poly_deg2", "zz_reps2_pairwise"
    kind: str           # "classical" | "quantum_fixed" | "quantum_learned" | "neural"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        valid_kinds = {"classical", "quantum_fixed", "quantum_learned", "neural"}
        if self.kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {self.kind!r}")


@dataclass(frozen=True)
class ModelConfig:
    """Regression model configuration."""
    name: str           # "ols" | "ridge" | "lasso" | "elasticnet"
    alpha_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    cv_folds: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""
    dgp: DGPConfig
    augmenters: list[AugmenterConfig]
    models: list[ModelConfig]
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    data_dir: str = "data/synthetic"
    results_dir: str = "results/synthetic"
    clip_range: float | None = 5.0  # Clip scaled features for quantum circuits (None = no clip)
