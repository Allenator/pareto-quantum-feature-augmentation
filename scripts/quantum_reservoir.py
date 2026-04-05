"""Quantum Reservoir Feature Augmentation

Standalone script that constructs and runs the optimal quantum feature map:
    RY encoding → [random Rot + CNOT]×L → random Rot → ⟨Zᵢ⟩

Multiple independent circuits (ensemble) with different random seeds produce
diverse features that are concatenated.

Usage:
    python scripts/quantum_reservoir.py                    # defaults
    python scripts/quantum_reservoir.py --n_qubits 6 --n_layers 3 --n_ensemble 5

As a library:
    from scripts.quantum_reservoir import QuantumReservoir
    qr = QuantumReservoir(n_qubits=4, n_layers=2, n_ensemble=3)
    features = qr.transform(X)  # X: (n_samples, n_qubits)
"""

import argparse

import numpy as np
import pennylane as qml


class QuantumReservoir:
    """Quantum reservoir feature augmentation via ensemble of random circuits.

    Circuit per ensemble member:
        RY(xᵢ) → [Rot(θ,φ,λ) + CNOT_linear]×n_layers → Rot(θ,φ,λ) → ⟨Zᵢ⟩

    Args:
        n_qubits: Number of qubits (must equal input feature dimension).
        n_layers: Number of [Rot + CNOT] mixing layers.
        n_ensemble: Number of independent circuits (different random seeds).
        seed: Base random seed. Circuit i uses seed + i.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 n_ensemble: int = 3, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_ensemble = n_ensemble
        self.seed = seed

        self.n_features_per_circuit = n_qubits
        self.n_features_total = n_ensemble * n_qubits
        self.n_random_params = n_ensemble * (n_layers + 1) * n_qubits * 3

        self._circuits = []
        self._all_weights = []
        self._build_circuits()

    def _build_circuits(self):
        n_q = self.n_qubits
        n_layers = self.n_layers
        pairs = [(i, i + 1) for i in range(n_q - 1)]

        for ens_idx in range(self.n_ensemble):
            rng = np.random.default_rng(self.seed + ens_idx)
            weights = rng.uniform(0, 2 * np.pi, (n_layers + 1, n_q, 3))
            self._all_weights.append(weights)

            dev = qml.device("lightning.qubit", wires=n_q)
            w = weights

            @qml.qnode(dev)
            def circuit(x, _w=w):
                # Angle encoding
                for i in range(n_q):
                    qml.RY(x[i], wires=i)
                # Mixing layers: random Rot + linear CNOT
                for layer in range(n_layers):
                    for i in range(n_q):
                        qml.Rot(_w[layer, i, 0], _w[layer, i, 1], _w[layer, i, 2], wires=i)
                    for i, j in pairs:
                        qml.CNOT(wires=[i, j])
                # Final random Rot before measurement
                for i in range(n_q):
                    qml.Rot(_w[n_layers, i, 0], _w[n_layers, i, 1], _w[n_layers, i, 2], wires=i)
                # Measure Z on each qubit
                return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

            self._circuits.append(circuit)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input features through the quantum reservoir ensemble.

        Args:
            X: Input array of shape (n_samples, n_qubits).

        Returns:
            Augmented features of shape (n_samples, n_ensemble * n_qubits).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"Input dimension {X.shape[1]} != n_qubits {self.n_qubits}"
            )

        all_features = []
        for circuit in self._circuits:
            feats = []
            for x in X:
                result = circuit(x)
                feats.append(np.array(result))
            all_features.append(np.array(feats))

        return np.hstack(all_features)

    def draw(self, x: np.ndarray | None = None) -> str:
        """Return text drawing of the first ensemble circuit."""
        if x is None:
            x = np.zeros(self.n_qubits)
        return qml.draw(self._circuits[0])(x)

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_ensemble={self.n_ensemble}, seed={self.seed})\n"
            f"  Features per circuit: {self.n_features_per_circuit}\n"
            f"  Total features: {self.n_features_total}\n"
            f"  Random parameters: {self.n_random_params}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Reservoir Feature Augmentation"
    )
    parser.add_argument("--n_qubits", type=int, default=4,
                        help="Number of qubits (= input feature dimension)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of [Rot + CNOT] mixing layers")
    parser.add_argument("--n_ensemble", type=int, default=3,
                        help="Number of independent circuits")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    qr = QuantumReservoir(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_ensemble=args.n_ensemble,
        seed=args.seed,
    )
    print(qr)
    print()

    # Demo with random input
    X = np.random.default_rng(0).standard_normal((5, args.n_qubits))
    print(f"Input shape: {X.shape}")

    features = qr.transform(X)
    print(f"Output shape: {features.shape}")
    print(f"\nFirst sample features:\n{features[0].round(4)}")

    print(f"\nCircuit (ensemble member 1):\n{qr.draw(X[0])}")


if __name__ == "__main__":
    main()
