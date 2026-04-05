"""Unified quantum feature augmenter with factorial design dimensions.

7 independent axes: encoding, connectivity, cnot_mixing, observables,
random_rot, n_layers, n_ensemble.
"""

import time

import numpy as np
import pennylane as qml

from src.synthetic.augmenters.base import AugmenterResult, _make_result


def _get_pairs(connectivity: str, n_qubits: int) -> list[tuple[int, int]]:
    if connectivity == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    elif connectivity == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 1:
            pairs.append((n_qubits - 1, 0))
        return pairs
    elif connectivity == "all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        raise ValueError(f"Unknown connectivity: {connectivity!r}")


def _build_measurements(wires: list[int], observables: str):
    n = len(wires)
    if observables == "Z":
        return [qml.expval(qml.PauliZ(w)) for w in wires]
    elif observables == "Z+ZZ":
        m = [qml.expval(qml.PauliZ(w)) for w in wires]
        m += [qml.expval(qml.PauliZ(wires[i]) @ qml.PauliZ(wires[j]))
              for i in range(n) for j in range(i + 1, n)]
        return m
    elif observables == "XYZ":
        m = []
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            m += [qml.expval(P(w)) for w in wires]
        return m
    elif observables == "full":
        m = []
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            m += [qml.expval(P(w)) for w in wires]
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            m += [qml.expval(P(wires[i]) @ P(wires[j]))
                  for i in range(n) for j in range(i + 1, n)]
        return m
    elif observables == "prob":
        return [qml.probs(wires=wires)]
    else:
        raise ValueError(f"Unknown observables: {observables!r}")


def _n_features(n_data: int, observables: str) -> int:
    n = n_data
    p = n * (n - 1) // 2
    return {
        "Z": n, "Z+ZZ": n + p, "XYZ": 3 * n,
        "full": 3 * n + 3 * p, "prob": 2 ** n,
    }[observables]


class UnifiedQuantumAugmenter:
    """Unified quantum feature map with 7 independent design dimensions.

    Circuit per ensemble member:
        [Encoding] → [Rot + CNOT]×n_layers → [Rot_final] → [Measure]

    Rot layers are present only if random_rot=True.
    CNOT layers are present only if cnot_mixing=True.
    """

    def __init__(self, encoding: str = "angle", connectivity: str = "linear",
                 cnot_mixing: bool = True, observables: str = "Z",
                 random_rot: bool = False, n_layers: int = 1,
                 n_ensemble: int = 1, seed: int = 42, **kwargs):
        self.encoding = encoding
        self.connectivity = connectivity
        self.cnot_mixing = cnot_mixing
        self.observables = observables
        self.random_rot = random_rot
        self.n_layers = n_layers
        self.n_ensemble = n_ensemble
        self.n_data = 4
        self.n_qubits = 4
        self._seed = seed

        # Name
        cnot_tag = "cnot" if cnot_mixing else "noc"
        obs_tag = f"_{observables}" if observables != "Z" else ""
        rot_tag = "_rot" if random_rot else ""
        lay_tag = f"_{n_layers}L" if n_layers != 1 else ""
        ens_tag = f"_{n_ensemble}ens" if n_ensemble != 1 else ""
        self.name = f"unified_{encoding}_{connectivity}_{cnot_tag}{obs_tag}{rot_tag}{lay_tag}{ens_tag}"

        # Random params
        if random_rot:
            self._n_random = n_ensemble * (n_layers + 1) * self.n_qubits * 3
        else:
            self._n_random = 0

        self._circuits = []
        self._all_weights = []
        self._specs_cache = None
        self._build_circuits()

    def _build_circuits(self):
        n_data = self.n_data
        n_q = self.n_qubits
        encoding = self.encoding
        connectivity = self.connectivity
        cnot_mixing = self.cnot_mixing
        observables = self.observables
        random_rot = self.random_rot
        n_layers = self.n_layers
        pairs = _get_pairs(connectivity, n_q)
        data_pairs = [(i, j) for i, j in pairs if i < n_data and j < n_data]
        data_wires = list(range(n_data))

        for ens_idx in range(self.n_ensemble):
            # Generate random weights for this ensemble member
            weights = None
            if random_rot:
                rng = np.random.default_rng(self._seed + ens_idx)
                weights = rng.uniform(0, 2 * np.pi, (n_layers + 1, n_q, 3))
            self._all_weights.append(weights)

            dev = qml.device("lightning.qubit", wires=n_q)
            w = weights  # capture for closure

            @qml.qnode(dev)
            def circuit(x, _w=w):
                # Encoding
                if encoding == "RZ":
                    for i in range(n_data):
                        qml.Hadamard(wires=i)
                        qml.RZ(x[i], wires=i)
                        qml.Hadamard(wires=i)
                elif encoding == "IQP":
                    for i in range(n_data):
                        qml.Hadamard(wires=i)
                        qml.RZ(x[i], wires=i)
                    for i, j in data_pairs:
                        qml.MultiRZ(x[i] * x[j], wires=[i, j])
                    for i in range(n_data):
                        qml.Hadamard(wires=i)
                elif encoding == "angle":
                    for i in range(n_data):
                        qml.RY(x[i], wires=i)

                # Mixing layers
                for layer in range(n_layers):
                    if random_rot and _w is not None:
                        for i in range(n_q):
                            qml.Rot(_w[layer, i, 0], _w[layer, i, 1], _w[layer, i, 2], wires=i)
                    if cnot_mixing:
                        for i, j in pairs:
                            qml.CNOT(wires=[i, j])

                # Final Rot before measurement
                if random_rot and _w is not None:
                    for i in range(n_q):
                        qml.Rot(_w[n_layers, i, 0], _w[n_layers, i, 1], _w[n_layers, i, 2], wires=i)

                return _build_measurements(data_wires, observables)

            self._circuits.append(circuit)

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuits[0])(x_sample)
            self._specs_cache = {
                "depth": specs.resources.depth,
                "gate_count": specs.resources.num_gates,
            }
        return self._specs_cache

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        specs = self._get_specs(X[0])
        is_prob = self.observables == "prob"

        all_features = []
        for circuit in self._circuits:
            feats = []
            for x in X:
                result = circuit(x)
                if is_prob:
                    feats.append(np.array(result[0]))
                else:
                    feats.append(np.array(result))
            all_features.append(np.array(feats))

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
