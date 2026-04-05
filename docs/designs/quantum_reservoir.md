# Quantum Reservoir Feature Augmentation

## Overview

The quantum reservoir is the optimal quantum feature augmentation strategy identified through systematic factorial evaluation of 1,620 circuit configurations across 7 independent design dimensions. It achieves the best complexity-MSE tradeoff among all quantum methods tested, and approaches or beats hand-crafted oracle features.

## Circuit Structure

Each ensemble member follows the template:

$$
|0\rangle^{\otimes n} \rightarrow{R_Y(x_i)} \rightarrow{[\text{Rot}(\theta,\phi,\lambda) + \text{CNOT}] \times L} \rightarrow{\text{Rot}(\theta,\phi,\lambda)} \rightarrow{\langle Z_i \rangle}
$$

Concretely for 4 qubits, 2 layers:

```
q0: ──RY(x₁)──Rot(θ₁)──●──Rot(θ₅)──●──Rot(θ₉)──┤ ⟨Z⟩
q1: ──RY(x₂)──Rot(θ₂)──X──●──Rot(θ₆)──X──●──Rot(θ₁₀)─┤ ⟨Z⟩
q2: ──RY(x₃)──Rot(θ₃)─────X──●──Rot(θ₇)─────X──●──Rot(θ₁₁)─┤ ⟨Z⟩
q3: ──RY(x₄)──Rot(θ₄)────────X──Rot(θ₈)────────X──Rot(θ₁₂)─┤ ⟨Z⟩
```

Where each `Rot(θ,φ,λ)` has 3 independent random parameters (fixed at initialization, not trained).

## Design Choices and Justification

Each design choice was validated by ablation against alternatives in the factorial sweep:

| Dimension | Choice | Alternatives tested | Rationale |
|-----------|--------|-------------------|-----------|
| **Encoding** | $R_Y$ (angle) | $R_Z$ (phase), IQP ($R_Z$ + ZZ) | $R_Y$ encodes data into amplitudes, directly visible in Z-basis measurement. $R_Z$/IQP encode into phases which are only accessible through interference — periodic wrapping degrades performance on heavy-tailed inputs. |
| **Connectivity** | Linear | Circular, all-to-all | <5% MSE difference across topologies. Linear is simplest and most hardware-friendly. |
| **CNOT mixing** | Yes | No | CNOTs create amplitude-level entanglement between qubits. Without CNOTs, each qubit evolves independently and pairwise features factorize. |
| **Observables** | Single Z ($\langle Z_i \rangle$) | Z+ZZ, XYZ, full, prob | Single Z produces 4 features per circuit — minimal per-circuit cost. Richer observables are better per-circuit but the ensemble scaling strategy (more circuits) is more parameter-efficient than richer measurements on fewer circuits. |
| **Random rotations** | Yes | No | Random `Rot` gates break the periodicity of the encoding and create diverse nonlinear features. Without randomization, bare $R_Y$ encoding gives $\langle Z_i \rangle = \cos(x_i)$ — useful but limited. Each random seed produces a different nonlinear projection. |
| **Mixing layers** | 2 (default) | 1, 3 | More layers increase circuit depth and random parameter count. 2 layers provides a good balance — 1 layer is too shallow for inter-qubit mixing, 3 layers adds marginal improvement at higher cost. |
| **Ensemble** | 3 (default) | 1, 2 | Multiple independent circuits with different random seeds produce diverse features. This is the primary scaling mechanism — equivalent to random Fourier features in classical ML, but through quantum nonlinear projections. |

## Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_qubits` | 4 | Number of qubits = input feature dimension. Features per circuit = `n_qubits`. |
| `n_layers` | 2 | Number of [Rot + CNOT] mixing blocks. Random params per circuit = `(n_layers + 1) × n_qubits × 3`. |
| `n_ensemble` | 3 | Number of independent circuits. Total features = `n_ensemble × n_qubits`. |
| `seed` | 42 | Base random seed. Circuit $i$ uses seed $+ i$. |

### Feature and parameter scaling

| n_qubits | n_layers | n_ensemble | Total features | Random params |
|----------|----------|-----------|---------------|---------------|
| 4 | 2 | 1 | 4 | 36 |
| 4 | 2 | 3 | 12 | 108 |
| 4 | 2 | 6 | 24 | 216 |
| 4 | 3 | 3 | 12 | 144 |
| 6 | 2 | 3 | 18 | 162 |
| 6 | 3 | 5 | 30 | 360 |

## Usage

### Command line

```bash
# Defaults (4 qubits, 2 layers, 3 ensemble → 12 features)
uv run python scripts/quantum_reservoir.py

# Custom
uv run python scripts/quantum_reservoir.py --n_qubits 6 --n_layers 3 --n_ensemble 5
```

### As a library

```python
from scripts.quantum_reservoir import QuantumReservoir
import numpy as np

qr = QuantumReservoir(n_qubits=4, n_layers=2, n_ensemble=3)
X = np.random.randn(100, 4)        # 100 samples, 4 features
features = qr.transform(X)         # (100, 12)

# Inspect circuit
print(qr)                          # summary
print(qr.draw(X[0]))               # text circuit diagram
```

### Integration with sklearn

```python
from scripts.quantum_reservoir import QuantumReservoir
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare data
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Augment
qr = QuantumReservoir(n_qubits=X_train.shape[1], n_layers=2, n_ensemble=3)
X_train_aug = np.hstack([X_train_s, qr.transform(X_train_s)])
X_test_aug = np.hstack([X_test_s, qr.transform(X_test_s)])

# Fit
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(X_train_aug, y_train)
predictions = model.predict(X_test_aug)
```

## Why It Works

The quantum reservoir's effectiveness comes from three properties:

1. **Nonlinear projection**: The $R_Y$ encoding followed by random rotations and entangling gates creates features that are complex nonlinear functions of the input — richer than polynomial features, different from RBF kernel features.

2. **Diversity via ensemble**: Each circuit with a different random seed produces a different nonlinear projection of the same input. The ensemble acts like a random feature expansion, but through quantum mechanics rather than classical random matrices.

3. **No training**: All parameters are random and fixed. This means zero overfitting risk from the augmenter itself, and the features are deterministic given the seed — fully reproducible.

## Relationship to Other Methods

| Method | Encoding | Entanglement | Parameters | Performance |
|--------|----------|-------------|------------|-------------|
| Polynomial (deg 2) | Deterministic | None | 0 | Good for interactions, fails on heavy tails |
| Random Fourier Features | cos(Wx+b) | None | Random W, b | Competitive at high feature count |
| **Quantum Reservoir** | **$R_Y$ + random Rot** | **CNOT** | **Random Rot angles** | **Best MSE per feature among quantum methods** |
| IQP | $R_Z$ + ZZ | Diagonal (phase) | 0 | Poor — periodic saturation |
| Trained MLP | Learned | None (classical) | Learned | Best absolute MSE, but requires training and is less interpretable |
