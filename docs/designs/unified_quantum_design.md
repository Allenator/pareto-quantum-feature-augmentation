# Unified Quantum Feature Map — Factorial Design

## Motivation

Early experiments tested quantum feature augmentation methods as separate, named circuits (ZZ map, IQP, angle encoding, reservoir, QAOA). Each had its own implementation with entangled design choices — encoding gates mixed with entanglement topology mixed with measurement strategy — making it impossible to attribute performance differences to any single factor.

Ablation studies revealed surprising results:
- Bare angle encoding ($R_Y$ only, no post-encoding layers) achieved MSE 2.06 — better than angle encoding with `StronglyEntanglingLayers` (MSE 2.49)
- IQP with or without a final Hadamard, with or without randomization, remained at MSE ~4.2–4.6 — fundamentally limited by diagonal circuit structure
- Entanglement topology (linear vs circular vs all-to-all) had <5% impact on MSE
- The reservoir's advantage came entirely from running **multiple independent circuits**, not circuit depth

These findings motivated a systematic refactoring: decompose the quantum circuit into independent design dimensions and sweep the full product space.

## Circuit Architecture

Every quantum feature map in the unified framework follows a single template:

$$
|0\rangle^{\otimes n} \xrightarrow{\text{Encoding}} \xrightarrow{[\text{Rot} + \text{CNOT}] \times L} \xrightarrow{\text{Rot}_{\text{final}}} \xrightarrow{\text{Measure}}
$$

The circuit is parameterized by 7 independent dimensions. Each ensemble member uses the same structure with a different random seed.

## Design Dimensions

### 1. Encoding (`encoding`)

How classical data $\mathbf{x} = (x_1, x_2, x_3, x_4)$ is embedded into the quantum state on 4 data qubits.

| Value | Circuit | Mechanism |
|-------|---------|-----------|
| `RZ` | $H \to R_Z(x_i) \to H$ per qubit | Phase encoding. Equivalent to $R_X(x_i)$. No entanglement from encoding. |
| `IQP` | $H \to R_Z(x_i) \to \text{MultiRZ}(x_i x_j) \to H$ | Phase encoding + data-dependent pairwise interaction. The ZZ pairs follow the `connectivity` topology. All data-dependent gates commute (diagonal circuit). |
| `angle` | $R_Y(x_i)$ per qubit | Amplitude encoding. Creates superposition $\cos(x_i/2)|0\rangle + \sin(x_i/2)|1\rangle$. No entanglement from encoding. |

**Key contrast**: IQP creates entanglement through data-dependent diagonal gates (phase correlations), while angle encoding relies on subsequent CNOT layers for entanglement (amplitude correlations). RZ is the non-interacting phase baseline.

### 2. Connectivity (`connectivity`)

Topology of qubit pairs used for both IQP's ZZ interactions and CNOT entangling gates.

| Value | Pairs (4 qubits) | Count |
|-------|-------------------|-------|
| `linear` | (0,1), (1,2), (2,3) | 3 |
| `circular` | (0,1), (1,2), (2,3), (3,0) | 4 |
| `all` | (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) | 6 |

### 3. CNOT Mixing (`cnot_mixing`)

Whether to apply a CNOT entangling layer after encoding (and after each random Rot layer, if enabled).

| Value | Effect |
|-------|--------|
| `False` | No CNOT gates. Each qubit evolves independently (unless IQP encoding provides ZZ interactions). |
| `True` | CNOT layer applied using the `connectivity` topology. Creates amplitude-level entanglement. |

**Independent from encoding**: IQP with `cnot_mixing=True` applies CNOTs *on top of* the ZZ interactions. Angle with `cnot_mixing=False` has no entanglement at all.

### 4. Observables (`observables`)

Measurement strategy on the 4 data qubits. Determines the number of features per circuit.

| Value | Measurements | Features |
|-------|-------------|----------|
| `Z` | $\langle Z_i \rangle$ | 4 |
| `Z+ZZ` | $\langle Z_i \rangle + \langle Z_i Z_j \rangle$ | 10 |
| `XYZ` | $\langle X_i \rangle + \langle Y_i \rangle + \langle Z_i \rangle$ | 12 |
| `full` | Single ($X$, $Y$, $Z$) + pairwise ($XX$, $YY$, $ZZ$) | 30 |
| `prob` | Computational basis probabilities $P(|b_1 b_2 b_3 b_4\rangle)$ | 16 |

### 5. Random Rotations (`random_rot`)

Whether to insert random $\text{Rot}(\theta, \phi, \lambda)$ gates on each qubit before each CNOT layer and before measurement.

| Value | Effect | Random params (4 qubits, $L$ layers) |
|-------|--------|--------------------------------------|
| `False` | No random rotations. Circuit is deterministic (given input). | 0 |
| `True` | Random Rot before each CNOT layer + final Rot before measurement. Each layer has independent random parameters. | $(L + 1) \times 4 \times 3$ |

Random rotations break the periodicity of deterministic encodings and create diverse feature mappings across ensemble members.

### 6. Number of Mixing Layers (`n_layers`)

Number of [Rot + CNOT] blocks after encoding. Each layer has its own independent random Rot parameters (if `random_rot=True`).

| Value | Circuit depth | Random params (if `random_rot=True`) |
|-------|-------------|--------------------------------------|
| 1 | Shallow | 24 |
| 2 | Medium | 36 |
| 3 | Deep | 48 |

### 7. Ensemble Size (`n_ensemble`)

Number of independent circuits with different random seeds. Features are concatenated.

| Value | Total features | Total random params |
|-------|---------------|---------------------|
| 1 | $1 \times f$ | $1 \times r$ |
| 2 | $2 \times f$ | $2 \times r$ |
| 3 | $3 \times f$ | $3 \times r$ |

Where $f$ is features per circuit (from `observables`) and $r$ is random params per circuit.

## Product Space

$$
|\text{Encoding}| \times |\text{Connectivity}| \times |\text{CNOT}| \times |\text{Observables}| \times |\text{Random}| \times |\text{Layers}| \times |\text{Ensemble}| = 3 \times 3 \times 2 \times 5 \times 2 \times 3 \times 3 = 1{,}620
$$

All 1,620 configurations are evaluated with Ridge regression (CV-tuned $\alpha$) on the same data split.

## Key Comparisons

The factorial design enables clean isolation of each factor's effect:

| Comparison | What it tests | Controlled variables |
|-----------|---------------|---------------------|
| IQP vs angle (both `cnot_mixing=False`) | Data-driven (ZZ) vs amplitude encoding, no entanglement from CNOT | connectivity, observables, rot, layers, ensemble |
| `cnot_mixing=True` vs `False` | Value of CNOT amplitude entanglement | encoding, connectivity, observables, rot, layers, ensemble |
| `random_rot=True` vs `False` | Value of random rotations | encoding, connectivity, cnot, observables, layers, ensemble |
| `n_layers` 1 vs 2 vs 3 | Depth scaling | all other dims fixed |
| `n_ensemble` 1 vs 2 vs 3 | Width scaling (more circuits) | all other dims fixed |
| `linear` vs `circular` vs `all` | Entanglement topology | encoding, cnot, observables, rot, layers, ensemble |
| Observable types | Information extraction strategy | encoding, connectivity, cnot, rot, layers, ensemble |

## Reference Baselines

Three horizontal reference lines are plotted on all figures:

| Baseline | MSE | Description |
|----------|-----|-------------|
| **Identity** | 4.77 | Raw 4 features, no augmentation |
| **Oracle** | 2.41 | Hand-crafted exact DGP terms: $X_1 X_3$ and $\log(|X_2| + 1)$ |
| **Noise floor** | 1.00 | $\text{Var}(\varepsilon)$ — irreducible error |

## Complexity Metrics

Each configuration is characterized by objective complexity metrics (no subjective rankings):

| Metric | What it measures |
|--------|-----------------|
| `n_trainable_params` | Learned parameters (0 for all static configs) |
| `n_random_params` | Fixed random parameters in Rot gates |
| `effective_rank` | Intrinsic dimensionality of augmented features (SVD entropy) |
| `nonlinearity_score` | Fraction of each feature not explained by linear function of input |
| `feature_target_alignment` | Average $|\rho(\tilde{x}_j, Y)|$ — individual feature predictiveness |
| `coef_l2_norm` | $||\hat{\beta}||_2$ of Ridge coefficients — model weight magnitude |

## Scaling Strategy

After identifying the best configuration from the factorial sweep, the winning method is scaled up along two axes:

1. **Depth**: increase `n_layers` from 1–3 to 1–5+
2. **Width**: increase `n_ensemble` from 1–3 to 1–12+

Feature count grows linearly with ensemble size. Performance is tracked against classical baselines (RFF, polynomial) at matched feature counts.
