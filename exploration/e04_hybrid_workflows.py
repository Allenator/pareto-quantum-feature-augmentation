"""
Section 6: Hybrid Quantum-Classical Workflows

Variational circuits as trainable feature transformers and quantum reservoir computing.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

N_QUBITS = 4
N_FEATURES = 4


# ---------------------------------------------------------------------------
# 6.1 Variational Quantum Circuit (VQC) as Feature Transformer
# ---------------------------------------------------------------------------
dev_vqc = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_vqc)
def vqc_feature_circuit(x, weights):
    """
    Variational circuit: alternating data-encoding and trainable layers.

    Args:
        x: input features, shape (N_FEATURES,)
        weights: trainable params, shape (n_layers, N_QUBITS, 3)
    """
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        # Data encoding
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        # Variational layer
        for i in range(N_QUBITS):
            qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
        # Entangling
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def train_vqc(X_train, y_train, n_layers=2, n_epochs=50, lr=0.1):
    """
    Train VQC parameters to minimize MSE proxy loss.

    Returns trained weights for use as feature extractor.
    """
    weights = pnp.random.uniform(0, 2 * np.pi, (n_layers, N_QUBITS, 3), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    def cost(weights, X_batch, y_batch):
        predictions = []
        for x in X_batch:
            out = vqc_feature_circuit(x, weights)
            predictions.append(out[0])  # Use first qubit expectation as proxy prediction
        predictions = pnp.array(predictions)
        return pnp.mean((predictions - y_batch) ** 2)

    for epoch in range(n_epochs):
        weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
        if (epoch + 1) % 10 == 0:
            c = cost(weights, X_train, y_train)
            print(f"  Epoch {epoch + 1}/{n_epochs}, Cost: {c:.6f}")

    return weights


# ---------------------------------------------------------------------------
# 6.2 Quantum Reservoir Computing
# ---------------------------------------------------------------------------
dev_reservoir = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_reservoir)
def reservoir_circuit(x, reservoir_weights):
    """
    Random fixed quantum circuit as nonlinear feature extractor.

    Args:
        x: input features, shape (N_FEATURES,)
        reservoir_weights: fixed random params, shape (n_layers, N_QUBITS, 3)
    """
    n_layers = reservoir_weights.shape[0]
    # Encode data
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")

    # Fixed random layers (not trained)
    for layer in range(n_layers):
        for i in range(N_QUBITS):
            qml.Rot(
                reservoir_weights[layer, i, 0],
                reservoir_weights[layer, i, 1],
                reservoir_weights[layer, i, 2],
                wires=i,
            )
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def generate_reservoir_features(X, n_reservoirs=3, n_layers=3, seed=42):
    """Generate features from multiple random reservoir circuits."""
    all_features = []
    for r in range(n_reservoirs):
        rng = np.random.default_rng(seed + r)
        reservoir_weights = rng.uniform(0, 2 * np.pi, (n_layers, N_QUBITS, 3))
        feats = []
        for x in X:
            result = reservoir_circuit(x, reservoir_weights)
            feats.append(np.array(result))
        all_features.append(np.array(feats))
    return np.hstack(all_features)


if __name__ == "__main__":
    from exploration.e01_data_generation import generate_regime_data

    X, Y, _ = generate_regime_data(200)
    X_small, y_small = X[:50], Y[:50]

    # VQC training
    print("=== VQC Training ===")
    trained_weights = train_vqc(X_small[:20], y_small[:20], n_layers=2, n_epochs=30, lr=0.05)

    vqc_feats = []
    for x in X_small[:5]:
        out = vqc_feature_circuit(x, trained_weights)
        vqc_feats.append(np.array(out))
    vqc_feats = np.array(vqc_feats)
    print(f"VQC features shape: {vqc_feats.shape}")

    # Reservoir computing
    print("\n=== Quantum Reservoir Computing ===")
    res_feats = generate_reservoir_features(X_small[:5], n_reservoirs=3, n_layers=3)
    print(f"Reservoir features shape: {res_feats.shape}")
