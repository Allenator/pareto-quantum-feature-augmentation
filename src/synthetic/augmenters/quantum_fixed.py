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


def _pad_features(x, n_qubits: int):
    """Cyclically repeat features to match n_qubits when n_qubits > len(x)."""
    n = len(x)
    if n >= n_qubits:
        return x[:n_qubits]
    reps = (n_qubits + n - 1) // n
    return np.tile(x, reps)[:n_qubits]


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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp, self._weights)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=int(np.prod(self._weights.shape)),
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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp)
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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )


def _build_observables(n_qubits: int, observables: str):
    """Build measurement list based on observable string.

    Supported values:
        "Z", "X", "Y" — single-Pauli on each qubit
        "XYZ" — all three single-Pauli on each qubit
        "Z+ZZ" — single Z + pairwise ZZ
        "XYZ+ZZ" — all three single + pairwise ZZ
        "full" — all single (X,Y,Z) + all pairwise (XX,YY,ZZ)
    """
    paulis = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    measurements = []

    if observables in ("Z", "X", "Y"):
        P = paulis[observables]
        measurements = [qml.expval(P(i)) for i in range(n_qubits)]

    elif observables == "XYZ":
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            measurements += [qml.expval(P(i)) for i in range(n_qubits)]

    elif observables == "Z+ZZ":
        measurements = list(_single_and_pairwise_z(n_qubits))

    elif observables == "XYZ+ZZ":
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            measurements += [qml.expval(P(i)) for i in range(n_qubits)]
        measurements += [
            qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
            for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]

    elif observables == "full":
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            measurements += [qml.expval(P(i)) for i in range(n_qubits)]
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            measurements += [
                qml.expval(P(i) @ P(j))
                for i in range(n_qubits) for j in range(i + 1, n_qubits)
            ]

    else:
        raise ValueError(f"Unknown observables: {observables!r}")

    return measurements


def _build_entangling_layer(n_qubits: int, entanglement: str):
    """Apply entangling gates based on topology string."""
    if entanglement == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif entanglement == "circular":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
    elif entanglement == "all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"Unknown entanglement: {entanglement!r}")


class ReservoirAugmenter:
    """Quantum reservoir computing: multiple fixed random circuits.

    Args:
        observables: Measurement basis — "Z", "X", "Y", "XYZ", "Z+ZZ", "XYZ+ZZ", "full"
        entanglement: CNOT topology — "linear", "circular", "all"
        data_reuploading: Re-encode data at every layer (not just the first)
        measure_pairwise: Deprecated, use observables="Z+ZZ" instead
    """

    def __init__(self, n_qubits: int = N_QUBITS_DEFAULT, n_reservoirs: int = 3,
                 n_layers: int = 3, observables: str = "Z",
                 entanglement: str = "linear", data_reuploading: bool = False,
                 measure_pairwise: bool = False, seed: int = 42):
        # Backward compat: measure_pairwise maps to "Z+ZZ"
        if measure_pairwise and observables == "Z":
            observables = "Z+ZZ"
        self.observables = observables
        self.entanglement = entanglement
        self.data_reuploading = data_reuploading

        obs_tag = f"_{observables}" if observables != "Z" else ""
        ent_tag = f"_{entanglement}" if entanglement != "linear" else ""
        reup_tag = "_reup" if data_reuploading else ""
        self.name = f"reservoir_{n_reservoirs}x{n_layers}{obs_tag}{ent_tag}{reup_tag}"

        self.n_qubits = n_qubits
        self.n_reservoirs = n_reservoirs
        self.n_layers = n_layers
        self._seed = seed
        self._circuits = []
        self._reservoir_weights = []
        self._specs_cache = None
        self._build_circuits()

    def _build_circuits(self):
        n_q = self.n_qubits
        obs = self.observables
        ent = self.entanglement
        reup = self.data_reuploading

        for r in range(self.n_reservoirs):
            rng = np.random.default_rng(self._seed + r)
            weights = rng.uniform(0, 2 * np.pi, (self.n_layers, n_q, 3))
            self._reservoir_weights.append(weights)

            dev = qml.device("lightning.qubit", wires=n_q)

            @qml.qnode(dev)
            def circuit(x, w):
                if not reup:
                    qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                for layer in range(w.shape[0]):
                    if reup:
                        qml.AngleEmbedding(x, wires=range(n_q), rotation="Y")
                    for i in range(n_q):
                        qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)
                    _build_entangling_layer(n_q, ent)
                return _build_observables(n_q, obs)

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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        all_features = []
        for r, (circuit, weights) in enumerate(zip(self._circuits, self._reservoir_weights)):
            feats = []
            for x in X:
                xp = _pad_features(x, self.n_qubits)
                result = circuit(xp, weights)
                feats.append(np.array(result))
            all_features.append(np.array(feats))
        X_new = np.hstack(all_features)
        elapsed = time.perf_counter() - t0
        n_random = sum(int(np.prod(w.shape)) for w in self._reservoir_weights)
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=n_random,
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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp, self._gammas, self._betas)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        n_random = len(self._gammas) + len(self._betas)
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=n_random,
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
        x0 = _pad_features(X[0], self.n_qubits)
        specs = self._get_specs(x0)
        features = []
        for x in X:
            xp = _pad_features(x, self.n_qubits)
            result = self._circuit(xp, self._weights)
            features.append(np.array(result))
        X_new = np.array(features)
        elapsed = time.perf_counter() - t0
        return _make_result(
            X, X_new, self.name, elapsed,
            n_random_params=int(np.prod(self._weights.shape)),
            circuit_depth=specs["depth"],
            qubit_count=self.n_qubits,
            gate_count=specs["gate_count"],
        )
