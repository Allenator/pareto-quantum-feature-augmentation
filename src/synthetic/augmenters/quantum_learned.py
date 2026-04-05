"""Learning-based quantum feature augmenters (VQC with trained weights).

By default circuits use lightning.qubit, but accept an optional ``device``
argument so callers can supply an AWS Braket device or other backend.
"""

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.synthetic.augmenters.base import AugmenterResult, _make_result
from src.synthetic.augmenters.quantum_fixed import _single_and_pairwise_z, _pad_features

N_QUBITS_DEFAULT = 4


class VQCAugmenter:
    """Variational Quantum Circuit with trained weights.

    Training uses a feature-learning objective: minimize MSE of a linear
    regression fit on the VQC features vs targets.
    """

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_layers: int = 2,
                 n_epochs: int = 200, lr: float = 0.01, batch_size: int = 64,
                 train_subsample: int | None = 500, seed: int = 42,
                 device=None):
        self.name = f"vqc_strong_{n_qubits}q_{n_layers}L"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_subsample = train_subsample
        self._seed = seed
        self._weights = None
        self._trained = False
        self._specs_cache = None
        self._dev = device if device is not None else qml.device("lightning.qubit", wires=n_qubits)
        # Adjoint diff only works on simulators; use parameter-shift for hardware
        self._diff_method = "adjoint"
        if device is not None and "braket" in getattr(device, "short_name", ""):
            self._diff_method = "parameter-shift"
        self._build_circuit()

    def _build_circuit(self):
        n_q = self.n_qubits

        @qml.qnode(self._dev, diff_method=self._diff_method)
        def circuit(x, weights):
            for layer in range(weights.shape[0]):
                qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                qml.StronglyEntanglingLayers(
                    weights[layer:layer+1], wires=range(n_q),
                )
            return _single_and_pairwise_z(n_q)

        self._circuit = circuit

    def _extract_features(self, X: np.ndarray, weights) -> np.ndarray:
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp, weights)
            features.append(np.array(result))
        return np.array(features)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        rng = np.random.default_rng(self._seed)

        # Subsample for tractable training
        if self.train_subsample and len(X_train) > self.train_subsample:
            idx = rng.choice(len(X_train), self.train_subsample, replace=False)
            X_sub = X_train[idx]
            y_sub = y_train[idx]
        else:
            X_sub = X_train
            y_sub = y_train

        # Initialize weights
        weights = pnp.array(
            rng.uniform(0, 2 * np.pi, (self.n_layers, self.n_qubits, 3)),
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(stepsize=self.lr)
        n_q = self.n_qubits

        n_samples = len(X_sub)
        for epoch in range(self.n_epochs):
            # Mini-batch
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                X_batch = X_sub[batch_idx]
                y_batch = y_sub[batch_idx]

                def cost(w):
                    feats = []
                    for x in X_batch:
                        xp = _pad_features(x, n_q)
                        result = self._circuit(xp, w)
                        feats.append(result)
                    feats = pnp.stack(feats)
                    predictions = pnp.mean(feats, axis=1)
                    return pnp.mean((predictions - y_batch) ** 2)

                weights = opt.step(cost, weights)

            if (epoch + 1) % 50 == 0:
                c = cost(weights)
                print(f"  VQC epoch {epoch + 1}/{self.n_epochs}, cost: {float(c):.4f}")

        self._weights = np.array(weights)
        self._trained = True

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            try:
                xp = _pad_features(x_sample, self.n_qubits)
                specs = qml.specs(self._circuit)(xp, self._weights)
                self._specs_cache = {
                    "depth": specs.resources.depth,
                    "gate_count": specs.resources.num_gates,
                }
            except Exception:
                self._specs_cache = {"depth": None, "gate_count": None}
        return self._specs_cache

    def transform(self, X: np.ndarray) -> AugmenterResult:
        if not self._trained:
            raise RuntimeError("VQCAugmenter.fit() must be called before transform()")
        t0 = time.perf_counter()
        specs = self._get_specs(X[0])
        X_new = self._extract_features(X, self._weights)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            n_trainable_params=int(np.prod(self._weights.shape)),
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )
