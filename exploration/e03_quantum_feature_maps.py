"""
Section 5: Quantum Feature Maps

Encode classical data into quantum states using rotation & entangling gates.
Covers angle encoding, ZZ feature map, and IQP encoding.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4
N_FEATURES = 4


# ---------------------------------------------------------------------------
# 5.1 Angle Encoding + Basic Entangler
# ---------------------------------------------------------------------------
dev_angle = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_angle)
def circuit_angle_basic_entangle(x, weights):
    """
    AngleEmbedding -> BasicEntanglerLayers -> Measurements.

    Args:
        x: input features, shape (N_FEATURES,)
        weights: trainable params for entangling layers, shape (n_layers, N_QUBITS)
    """
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))

    single = [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    pairs = [
        qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
        for i in range(N_QUBITS)
        for j in range(i + 1, N_QUBITS)
    ]
    return single + pairs


# ---------------------------------------------------------------------------
# 5.2 ZZ Feature Map
# ---------------------------------------------------------------------------
dev_zz = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_zz)
def circuit_zz_feature_map(x, reps=2):
    """
    ZZ feature map: Hadamard + Rz encoding + ZZ entangling interactions.

    Args:
        x: input features, shape (N_FEATURES,)
        reps: number of repetitions of the encoding + entangling block
    """
    for _ in range(reps):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                qml.CNOT(wires=[i, j])
                qml.RZ(x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# 5.3 IQP Encoding
# ---------------------------------------------------------------------------
dev_iqp = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_iqp)
def circuit_iqp_encoding(x, n_repeats=2):
    """
    IQP encoding using PennyLane's IQPEmbedding template.

    Args:
        x: input features, shape (N_FEATURES,)
        n_repeats: number of repetitions
    """
    qml.IQPEmbedding(x, wires=range(N_QUBITS), n_repeats=n_repeats)
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def extract_quantum_features(circuit_fn, X, **circuit_kwargs):
    """Run a quantum circuit on each sample and collect features."""
    features = []
    for x in X:
        result = circuit_fn(x, **circuit_kwargs)
        features.append(np.array(result))
    return np.array(features)


if __name__ == "__main__":
    from exploration.e01_data_generation import generate_regime_data

    X, Y, _ = generate_regime_data(100)

    # Test angle encoding
    n_layers = 2
    weights = pnp.random.uniform(0, 2 * np.pi, (n_layers, N_QUBITS), requires_grad=True)
    feats_angle = extract_quantum_features(
        circuit_angle_basic_entangle, X[:5], weights=weights
    )
    print(f"Angle encoding features shape: {feats_angle.shape}")
    print(f"Sample output: {feats_angle[0]}")

    # Test ZZ feature map
    feats_zz = extract_quantum_features(circuit_zz_feature_map, X[:5], reps=2)
    print(f"\nZZ feature map shape: {feats_zz.shape}")
    print(f"Sample output: {feats_zz[0]}")

    # Test IQP encoding
    feats_iqp = extract_quantum_features(circuit_iqp_encoding, X[:5], n_repeats=2)
    print(f"\nIQP encoding shape: {feats_iqp.shape}")
    print(f"Sample output: {feats_iqp[0]}")
