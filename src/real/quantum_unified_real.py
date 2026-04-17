"""Generalized unified quantum reservoir augmenter for real data.

Adapts the winning synthetic augmenter design (angle encoding, circular CNOT,
random rotations, configurable observables) to arbitrary input feature dimensions
with feature-to-qubit mapping strategies and parameter broadcasting.
"""

import time

import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA

from src.synthetic.augmenters.base import AugmenterResult, _make_result
from src.synthetic.augmenters.quantum_unified import (
    _build_measurements,
    _get_pairs,
    _n_features,
)


class UnifiedReservoirAugmenter:
    """Unified quantum reservoir augmenter with variable input dimension.

    Generalizes UnifiedQuantumAugmenter (synthetic, hardcoded n=4) to handle
    arbitrary n_features via feature-to-qubit mapping strategies.

    Circuit per ensemble member:
        [Map features] → [Encoding] → [Rot + CNOT]×n_layers → [Rot_final] → [Measure]

    Qubit mapping strategies:
        - "direct": 1:1 mapping, n_qubits must equal n_features
        - "modular": average features into n_qubits bins (modular assignment)
        - "pca": PCA reduction to n_qubits dimensions (requires fit())
        - "cyclic": re-upload different feature slices at each layer — no information
          loss, uses circuit depth to accommodate all features
    """

    def __init__(self, n_features: int, n_qubits: int | None = None,
                 encoding: str = "angle", connectivity: str = "circular",
                 cnot_mixing: bool = True, observables: str = "Z",
                 random_rot: bool = True, n_layers: int = 3,
                 n_ensemble: int = 3, qubit_mapping: str = "modular",
                 seed: int = 42, **kwargs):
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits is not None else n_features
        self.encoding = encoding
        self.connectivity = connectivity
        self.cnot_mixing = cnot_mixing
        self.observables = observables
        self.random_rot = random_rot
        self.n_layers = n_layers
        self.n_ensemble = n_ensemble
        self.qubit_mapping = qubit_mapping
        self._seed = seed

        if qubit_mapping == "direct" and self.n_qubits != n_features:
            raise ValueError(
                f"direct mapping requires n_qubits == n_features, "
                f"got {self.n_qubits} != {n_features}"
            )

        # Name
        cnot_tag = "cnot" if cnot_mixing else "noc"
        obs_tag = f"_{observables}" if observables != "Z" else ""
        rot_tag = "_rot" if random_rot else ""
        lay_tag = f"_{n_layers}L" if n_layers != 1 else ""
        ens_tag = f"_{n_ensemble}ens" if n_ensemble != 1 else ""
        map_tag = f"_{qubit_mapping}" if qubit_mapping != "modular" else ""
        self.name = (f"qunified_{encoding}_{connectivity}_{cnot_tag}"
                     f"{obs_tag}{rot_tag}{lay_tag}{ens_tag}"
                     f"_{self.n_qubits}q{map_tag}")

        # Random params
        if random_rot:
            self._n_random = n_ensemble * (n_layers + 1) * self.n_qubits * 3
        else:
            self._n_random = 0

        # PCA state (fitted in fit() if qubit_mapping == "pca")
        self._pca = None

        # Modular mapping: precompute bin assignments
        if qubit_mapping == "modular" and self.n_qubits < n_features:
            self._bin_indices = [[] for _ in range(self.n_qubits)]
            for i in range(n_features):
                self._bin_indices[i % self.n_qubits].append(i)

        # Cyclic mapping: precompute per-layer feature slices
        if qubit_mapping == "cyclic":
            self._cyclic_slices = []
            for layer in range(n_layers + 1):  # +1 for initial encoding
                start = (layer * self.n_qubits) % n_features
                indices = [(start + j) % n_features for j in range(self.n_qubits)]
                self._cyclic_slices.append(indices)

        self._circuits = []
        self._all_weights = []
        self._specs_cache = None
        self._build_circuits()

    def _map_features(self, X: np.ndarray) -> np.ndarray:
        """Map n_features input to n_qubits circuit input.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            shape (n_samples, n_qubits)
        """
        if self.qubit_mapping == "direct":
            return X

        if self.qubit_mapping == "pca":
            if self._pca is None:
                raise RuntimeError("PCA mapping requires fit() before transform()")
            return self._pca.transform(X)

        # modular: average features assigned to each qubit bin
        if self.n_qubits >= self.n_features:
            # No compression needed — pad with zeros if n_qubits > n_features
            if self.n_qubits == self.n_features:
                return X
            result = np.zeros((X.shape[0], self.n_qubits))
            result[:, :self.n_features] = X
            return result

        result = np.zeros((X.shape[0], self.n_qubits))
        for j, indices in enumerate(self._bin_indices):
            result[:, j] = X[:, indices].mean(axis=1)
        return result

    def _build_circuits(self):
        n_q = self.n_qubits
        encoding = self.encoding
        connectivity = self.connectivity
        cnot_mixing = self.cnot_mixing
        observables = self.observables
        random_rot = self.random_rot
        n_layers = self.n_layers
        is_cyclic = self.qubit_mapping == "cyclic"
        pairs = _get_pairs(connectivity, n_q)
        meas_wires = list(range(n_q))

        # For cyclic: capture the per-layer feature index slices
        cyclic_slices = self._cyclic_slices if is_cyclic else None

        for ens_idx in range(self.n_ensemble):
            weights = None
            if random_rot:
                rng = np.random.default_rng(self._seed + ens_idx)
                weights = rng.uniform(0, 2 * np.pi, (n_layers + 1, n_q, 3))
            self._all_weights.append(weights)

            dev = qml.device("lightning.qubit", wires=n_q)
            w = weights  # capture for closure
            _slices = cyclic_slices  # capture for closure

            def _encode(x, feat_indices=None):
                """Encode features into qubits. For cyclic, x is the full feature
                vector and feat_indices selects which features to encode."""
                if feat_indices is not None:
                    # Cyclic: select features for this layer
                    x_slice = x[..., feat_indices]
                else:
                    x_slice = x

                if encoding == "angle":
                    qml.AngleEmbedding(x_slice, wires=range(n_q), rotation="Y")
                elif encoding == "RZ":
                    for i in range(n_q):
                        qml.Hadamard(wires=i)
                        qml.RZ(x_slice[..., i], wires=i)
                        qml.Hadamard(wires=i)
                elif encoding == "IQP":
                    for i in range(n_q):
                        qml.Hadamard(wires=i)
                        qml.RZ(x_slice[..., i], wires=i)
                    data_pairs = [(i, j) for i, j in pairs if i < n_q and j < n_q]
                    for i, j in data_pairs:
                        qml.MultiRZ(x_slice[..., i] * x_slice[..., j], wires=[i, j])
                    for i in range(n_q):
                        qml.Hadamard(wires=i)

            @qml.qnode(dev)
            def circuit(x, _w=w, _sl=_slices):
                # Initial encoding
                _encode(x, feat_indices=_sl[0] if _sl else None)

                # Mixing layers with optional re-encoding
                for layer in range(n_layers):
                    if random_rot and _w is not None:
                        for i in range(n_q):
                            qml.Rot(_w[layer, i, 0], _w[layer, i, 1],
                                    _w[layer, i, 2], wires=i)
                    if cnot_mixing:
                        for i, j in pairs:
                            qml.CNOT(wires=[i, j])
                    # Cyclic re-upload: encode next feature slice after mixing
                    if _sl is not None:
                        _encode(x, feat_indices=_sl[layer + 1])

                # Final Rot before measurement
                if random_rot and _w is not None:
                    for i in range(n_q):
                        qml.Rot(_w[n_layers, i, 0], _w[n_layers, i, 1],
                                _w[n_layers, i, 2], wires=i)

                return _build_measurements(meas_wires, observables)

            self._circuits.append(circuit)

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            try:
                specs = qml.specs(self._circuits[0])(x_sample)
                self._specs_cache = {
                    "depth": specs.resources.depth,
                    "gate_count": specs.resources.num_gates,
                }
            except Exception:
                self._specs_cache = {"depth": None, "gate_count": None}
        return self._specs_cache

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the augmenter. Only PCA mapping requires fitting."""
        if self.qubit_mapping == "pca":
            self._pca = PCA(n_components=self.n_qubits)
            self._pca.fit(X_train)

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()

        # For cyclic mapping, pass raw features — the circuit slices internally.
        # For all other mappings, pre-map to n_qubits dimensions.
        if self.qubit_mapping == "cyclic":
            X_circuit = X  # (n_samples, n_features)
        else:
            X_circuit = self._map_features(X)  # (n_samples, n_qubits)

        specs = self._get_specs(X_circuit[0])
        is_prob = self.observables == "prob"

        all_features = []
        for circuit in self._circuits:
            # Parameter broadcasting: pass full batch at once
            result = circuit(X_circuit)
            if is_prob:
                # probs returns (n_samples, 2^n_qubits) directly
                all_features.append(np.array(result[0]))
            else:
                # Each observable returns (n_samples,), stack them
                all_features.append(np.stack(result, axis=-1))

        if self.n_ensemble == 1:
            X_new = all_features[0]
        else:
            X_new = np.hstack(all_features)

        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=self._n_random,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )
