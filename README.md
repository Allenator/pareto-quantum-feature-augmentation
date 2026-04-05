# Pareto-Optimal Quantum Feature Augmentation

**YQuantum 2026 — AWS x State Street Challenge**

> Do quantum-derived feature transformations improve out-of-sample financial prediction relative to classical feature engineering?

We answer this through a systematic factorial evaluation of 1,620 quantum circuit configurations across 7 independent design dimensions, compared against classical baselines under strict fairness controls.

## Key Result

![Pareto Frontier: Quantum vs Classical](plots/synthetic/pareto_vs_classical/pareto_vs_classical.png)

Quantum reservoir methods achieve the **Pareto-optimal MSE-complexity tradeoff**, matching or beating the best classical baselines (Random Fourier Features) at every feature count — with lower cross-seed variance.

| Method | Features | MSE | Correlation |
|--------|----------|-----|-------------|
| **Quantum Pareto** (best) | 94 | **1.38 ± 0.15** | **0.980** |
| RFF 96 (best classical) | 100 | 1.45 ± 0.26 | 0.979 |
| Oracle (exact DGP terms) | 6 | 2.41 ± 0.02 | 0.965 |
| Identity (no augmentation) | 4 | 4.68 ± 0.06 | 0.931 |

## Unified Circuit Design

All quantum feature maps are instances of a single parameterized template:

![Unified Circuit Template](plots/illustrations/8_summary.png)

### 7 Design Dimensions

| Dimension | Values | Effect |
|-----------|--------|--------|
| **Encoding** | RZ, IQP, angle | How data enters the circuit |
| **Connectivity** | linear, circular, all | Which qubits interact |
| **CNOT mixing** | yes / no | Amplitude entanglement |
| **Observables** | Z, Z+ZZ, XYZ, full, prob | What is measured |
| **Random rotations** | yes / no | Breaks encoding periodicity |
| **Mixing layers** | 1, 2, 3 | Circuit depth |
| **Ensemble size** | 1, 2, 3 | Independent circuit count |

Full product space: $3 \times 3 \times 2 \times 5 \times 2 \times 3 \times 3 = 1{,}620$ configurations.

### Optimal Configuration: Quantum Reservoir

The Pareto-optimal strategy identified through the factorial sweep:

```
RY(xᵢ) → [random Rot + CNOT]×L → random Rot → ⟨Zᵢ⟩    (× n_ensemble circuits)
```

![Reservoir Circuit](plots/illustrations/circuit_reservoir.png)

- **Angle encoding** ($R_Y$) — amplitude-based, robust to heavy tails
- **Random rotations** — break periodicity, create diverse nonlinear features
- **Linear CNOT connectivity** — sufficient entanglement, hardware-friendly
- **Single-Z measurement** — minimal per-circuit cost, scale via ensemble
- **Ensemble of independent circuits** — primary scaling mechanism

### Factorial Analysis

![Encoding × Structure](plots/synthetic/unified_factorial/figures/encoding_x_structure.png)

Key findings from the 2×2 encoding-structure comparison:
- **Angle encoding dominates IQP** across all structures (amplitude vs phase encoding)
- **Random rotations are essential** — without them, performance degrades regardless of encoding
- **CNOT mixing helps only with randomization** — alone it can hurt due to destructive interference

## Classical Baselines

| Method | Features | MSE | Description |
|--------|----------|-----|-------------|
| **RFF** (6–96 components) | 10–100 | 2.48–1.45 | Random cosine projections (RBF kernel) |
| **Polynomial** (deg 2) | 14 | 2.88 ± 1.03 | All pairwise interactions + squares |
| **Interaction + Log/Abs** | 18 | 2.62 ± 0.66 | Hand-crafted nonlinear transforms |
| **Oracle** | 6 | 2.41 ± 0.02 | Exact DGP terms ($X_1 X_3$, $\log|X_2|$) |
| **Identity** | 4 | 4.68 ± 0.06 | Raw features, no augmentation |

RFF is the strongest classical competitor — both RFF and quantum reservoir use random projections, but through different mechanisms (classical cosine vs quantum circuit).

## Overfitting Diagnostic

![Overfitting Diagnostic](plots/synthetic/pareto_vs_classical/overfitting_diagnostic.png)

All methods show minimal overfitting (test ≈ train MSE). Quantum methods have tighter train-test gaps than polynomial baselines, which suffer from Cauchy-tail instability.

## Complexity Analysis

![Performance vs Random Parameters](plots/synthetic/pareto_vs_classical/params_vs_performance.png)

Quantum reservoir achieves better MSE per random parameter than RFF — the quantum nonlinear projection is more parameter-efficient than classical cosine projections.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running Experiments

```bash
# Unified factorial sweep (1,620 quantum configs)
uv run python scripts/run_synthetic.py unified

# Static augmenter sweep (classical + legacy quantum)
uv run python scripts/run_synthetic.py static

# Pareto frontier vs classical baselines on full data
# (automatically selects Pareto-optimal configs)

# Generate interactive plots
uv run python scripts/plot_unified.py
uv run python scripts/plot_pareto_vs_classical.py

# Standalone quantum reservoir
uv run python scripts/quantum_reservoir.py --n_qubits 4 --n_layers 2 --n_ensemble 3
```

## Project Structure

```
src/synthetic/             # Modular experiment framework
  augmenters/
    quantum_unified.py     # Unified 7-dimension quantum augmenter
    quantum_fixed.py       # Legacy quantum augmenters (angle, ZZ, IQP, reservoir, QAOA)
    classical.py           # Polynomial, RFF, oracle, log/abs
    neural.py              # MLP, autoencoder, learned RFF (trainable)
  models/linear.py         # OLS, RidgeCV, LassoCV, ElasticNetCV
  evaluation/
    metrics.py             # MSE, MAE, correlation + per-regime + train metrics
    complexity.py          # Effective rank, nonlinearity, feature-target alignment
    comparison.py          # Fairness checking, result aggregation
  runner.py                # Multiprocessing orchestrator with tqdm, caching, feature saving
  config.py                # Frozen dataclasses for all configuration
  dgp.py                   # Synthetic regime-switching DGP
scripts/
  run_synthetic.py         # Experiment entry point (unified / static / trainable)
  plot_unified.py          # Factorial sweep interactive plots
  plot_pareto_vs_classical.py  # Pareto frontier comparison plots
  quantum_reservoir.py     # Standalone quantum reservoir (library + CLI)
docs/designs/              # Design documents and illustrations
plots/                     # Generated figures (interactive HTML + PNG)
results/                   # Experiment results (JSON + summary CSV)
```

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for full documentation index:
- [Challenge Specification](docs/specs/challenge.md)
- [Unified Quantum Design](docs/designs/unified_quantum_design.md) — Factorial design strategy
- [Quantum Reservoir](docs/designs/quantum_reservoir.md) — Optimal circuit design and usage
- [Classical Baselines](docs/designs/classical_baselines.md) — Classical methods and comparison

## Challenge Reference

See [AWS-State-Street-Challenge/README.md](AWS-State-Street-Challenge/README.md) for the original challenge specification.
