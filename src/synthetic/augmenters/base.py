"""FeatureAugmenter protocol and AugmenterResult dataclass."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class AugmenterResult:
    """Output of a feature augmenter, carrying features + metadata."""
    features: np.ndarray        # (n_samples, n_original + n_augmented)
    n_original: int
    n_augmented: int
    augmenter_name: str
    wall_clock_seconds: float
    # Quantum-specific (None for classical)
    circuit_depth: int | None = None
    qubit_count: int | None = None
    gate_count: int | None = None


@runtime_checkable
class FeatureAugmenter(Protocol):
    """Protocol for all feature augmenters (classical and quantum)."""
    name: str

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Learn parameters if needed. No-op for fixed augmenters."""
        ...

    def transform(self, X: np.ndarray) -> AugmenterResult:
        """Produce augmented feature matrix [X_original | X_new]."""
        ...


def _make_result(
    X_original: np.ndarray,
    X_new: np.ndarray | None,
    name: str,
    elapsed: float,
    circuit_depth: int | None = None,
    qubit_count: int | None = None,
    gate_count: int | None = None,
) -> AugmenterResult:
    """Helper to build AugmenterResult with original features prepended."""
    if X_new is not None:
        features = np.hstack([X_original, X_new])
        n_augmented = X_new.shape[1]
    else:
        features = X_original.copy()
        n_augmented = 0
    return AugmenterResult(
        features=features,
        n_original=X_original.shape[1],
        n_augmented=n_augmented,
        augmenter_name=name,
        wall_clock_seconds=elapsed,
        circuit_depth=circuit_depth,
        qubit_count=qubit_count,
        gate_count=gate_count,
    )
