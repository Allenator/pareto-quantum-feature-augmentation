"""Learning-based quantum feature augmenters (VQC with trained weights)."""

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.synthetic.augmenters.base import AugmenterResult, _make_result
from src.synthetic.augmenters.quantum_fixed import _single_and_pairwise_z

N_QUBITS_DEFAULT = 4


class VQCAugmenter:
    """Variational Quantum Circuit with trained weights.

    Training uses a feature-learning objective: minimize MSE of a linear
    regression fit on the VQC features vs targets.
    """

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_layers: int = 2,
                 n_epochs: int = 200, lr: float = 0.01, batch_size: int = 64,
                 train_subsample: int | None = 500, seed: int = 42):
        self.name = f"vqc_strong_{n_layers}L"
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
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        n_q = self.n_qubits

        @qml.qnode(self._dev, diff_method="adjoint")
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
            result = self._circuit(x, weights)
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
                        result = self._circuit(x, w)
                        feats.append(result)
                    feats = pnp.stack(feats)
                    # Least-squares proxy: mean squared feature-target correlation
                    # Use sum of squared correlations as proxy
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
            specs = qml.specs(self._circuit)(x_sample, self._weights)
            self._specs_cache = {
                "depth": specs.resources.depth,
                "gate_count": specs.resources.num_gates,
            }
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
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )
