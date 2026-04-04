"""Fixed-parameter quantum feature augmenters.

All circuits use lightning.qubit for performance.
All produce [X_original | X_quantum] with quantum resource metadata.
"""

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.synthetic.augmenters.base import AugmenterResult, _make_result

N_QUBITS_DEFAULT = 4


def _single_and_pairwise_z(n_qubits: int):
    """Return measurement list: single Z + pairwise ZZ expectations."""
    single = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    pairs = [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
        for i in range(n_qubits) for j in range(i + 1, n_qubits)
    ]
    return single + pairs


def _n_features_z_zz(n_qubits: int) -> int:
    return n_qubits + n_qubits * (n_qubits - 1) // 2


class AngleEncodingAugmenter:
    """Angle encoding + entangling layers with Z + ZZ observables."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_layers: int = 2,
                 use_strongly_entangling: bool = False, seed: int = 42):
        suffix = "strong" if use_strongly_entangling else "basic"
        self.name = f"angle_{suffix}_{n_layers}L"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_strongly_entangling = use_strongly_entangling
        self._seed = seed
        self._weights = None
        self._circuit = None
        self._specs_cache = None
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        n_q, n_l = self.n_qubits, self.n_layers
        rng = np.random.default_rng(self._seed)

        if self.use_strongly_entangling:
            self._weights = rng.uniform(0, 2 * np.pi, (n_l, n_q, 3))
            @qml.qnode(self._dev)
            def circuit(x, weights):
                qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_q))
                return _single_and_pairwise_z(n_q)
        else:
            self._weights = rng.uniform(0, 2 * np.pi, (n_l, n_q))
            @qml.qnode(self._dev)
            def circuit(x, weights):
                qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                qml.BasicEntanglerLayers(weights, wires=range(n_q))
                return _single_and_pairwise_z(n_q)

        self._circuit = circuit

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuit)(x_sample, self._weights)
            self._specs_cache = {
                "depth": specs.resources.depth,
                "gate_count": specs.resources.num_gates,
            }
        return self._specs_cache

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass  # Fixed weights

    def transform(self, X: np.ndarray) -> AugmenterResult:
        t0 = time.perf_counter()
        specs = self._get_specs(X[0])
        features = []
        for x in X:
            result = self._circuit(x, self._weights)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


class ZZMapAugmenter:
    """ZZ feature map with Z + ZZ pairwise observables (bug-fixed)."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, reps: int = 2):
        self.name = f"zz_reps{reps}"
        self.n_qubits = n_qubits
        self.reps = reps
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._specs_cache = None
        self._build_circuit()

    def _build_circuit(self):
        n_q, reps = self.n_qubits, self.reps

        @qml.qnode(self._dev)
        def circuit(x):
            for _ in range(reps):
                for i in range(n_q):
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)
                for i in range(n_q):
                    for j in range(i + 1, n_q):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(x[i] * x[j], wires=j)
                        qml.CNOT(wires=[i, j])
            return _single_and_pairwise_z(n_q)

        self._circuit = circuit

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuit)(x_sample)
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
        features = []
        for x in X:
            result = self._circuit(x)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


class IQPAugmenter:
    """IQP encoding, differentiated from ZZ by using n_repeats=3 and Z+ZZ obs."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_repeats: int = 3):
        self.name = f"iqp_reps{n_repeats}"
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._specs_cache = None
        self._build_circuit()

    def _build_circuit(self):
        n_q, n_r = self.n_qubits, self.n_repeats

        @qml.qnode(self._dev)
        def circuit(x):
            qml.IQPEmbedding(x, wires=range(n_q), n_repeats=n_r)
            return _single_and_pairwise_z(n_q)

        self._circuit = circuit

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuit)(x_sample)
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
        features = []
        for x in X:
            result = self._circuit(x)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


class ReservoirAugmenter:
    """Quantum reservoir computing: multiple fixed random circuits."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_reservoirs: int = 3,
                 n_layers: int = 3, measure_pairwise: bool = False, seed: int = 42):
        pairwise_tag = "_zz" if measure_pairwise else ""
        self.name = f"reservoir_{n_reservoirs}x{n_layers}{pairwise_tag}"
        self.n_qubits = n_qubits
        self.n_reservoirs = n_reservoirs
        self.n_layers = n_layers
        self.measure_pairwise = measure_pairwise
        self._seed = seed
        self._circuits = []
        self._reservoir_weights = []
        self._specs_cache = None
        self._build_circuits()

    def _build_circuits(self):
        n_q = self.n_qubits
        for r in range(self.n_reservoirs):
            rng = np.random.default_rng(self._seed + r)
            weights = rng.uniform(0, 2 * np.pi, (self.n_layers, n_q, 3))
            self._reservoir_weights.append(weights)

            dev = qml.device("lightning.qubit", wires=n_q)

            @qml.qnode(dev)
            def circuit(x, w):
                qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                for layer in range(w.shape[0]):
                    for i in range(n_q):
                        qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)
                    for i in range(n_q - 1):
                        qml.CNOT(wires=[i, i + 1])
                if self.measure_pairwise:
                    return _single_and_pairwise_z(n_q)
                else:
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

            self._circuits.append(circuit)

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuits[0])(x_sample, self._reservoir_weights[0])
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
        all_features = []
        for r, (circuit, weights) in enumerate(zip(self._circuits, self._reservoir_weights)):
            feats = []
            for x in X:
                result = circuit(x, weights)
                feats.append(np.array(result))
            all_features.append(np.array(feats))
        X_new = np.hstack(all_features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


class QAOAAugmenter:
    """QAOA-inspired feature map: alternating cost(x)/mixer Hamiltonians."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, p: int = 2, seed: int = 42):
        self.name = f"qaoa_p{p}"
        self.n_qubits = n_qubits
        self.p = p
        self._seed = seed
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._specs_cache = None
        rng = np.random.default_rng(seed)
        self._gammas = rng.uniform(0.1, 1.0, p)
        self._betas = rng.uniform(0.1, 1.0, p)
        self._build_circuit()

    def _build_circuit(self):
        n_q = self.n_qubits

        @qml.qnode(self._dev)
        def circuit(x, gammas, betas):
            # Initial superposition
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for layer in range(len(gammas)):
                # Cost Hamiltonian: encode data
                for i in range(n_q):
                    qml.RZ(gammas[layer] * x[i], wires=i)
                for i in range(n_q):
                    for j in range(i + 1, n_q):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(gammas[layer] * x[i] * x[j], wires=j)
                        qml.CNOT(wires=[i, j])
                # Mixer Hamiltonian
                for i in range(n_q):
                    qml.RX(betas[layer], wires=i)
            return _single_and_pairwise_z(n_q)

        self._circuit = circuit

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuit)(x_sample, self._gammas, self._betas)
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
        features = []
        for x in X:
            result = self._circuit(x, self._gammas, self._betas)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


class ProbabilityAugmenter:
    """Angle encoding + StronglyEntanglingLayers with probability extraction."""

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_layers: int = 2, seed: int = 42):
        self.name = f"angle_strong_{n_layers}L_prob"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self._seed = seed
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._specs_cache = None
        rng = np.random.default_rng(seed)
        self._weights = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        self._build_circuit()

    def _build_circuit(self):
        n_q = self.n_qubits

        @qml.qnode(self._dev)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_q))
            return qml.probs(wires=range(n_q))

        self._circuit = circuit

    def _get_specs(self, x_sample):
        if self._specs_cache is None:
            specs = qml.specs(self._circuit)(x_sample, self._weights)
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
        features = []
        for x in X:
            result = self._circuit(x, self._weights)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )
