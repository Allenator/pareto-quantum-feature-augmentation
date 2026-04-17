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

![Unified Circuit Template](plots/illustrations/architecture/8_summary.png)

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

![Reservoir Circuit](plots/illustrations/architecture/circuit_reservoir.png)

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

## Part II — Real Financial Data

### S&P 500 Excess Return Prediction

The same quantum reservoir design is applied to predicting 5-day forward excess returns of 10 S&P 500 stocks (vs SPY), using walk-forward backtesting with a rolling 504 trading-day window.

![Walk-Forward Timeline](plots/illustrations/real/1_walk_forward_timeline.png)

**Data pipeline:** DataBento OHLCV (XNAS.ITCH, 2022–2025), 17 per-stock features (14 stock-minus-market + 3 cross-asset regime indicators), plus quantum-encoded correlation matrix features. Automatic stock split detection and adjustment.

![Feature Construction Pipeline](plots/illustrations/real/2_feature_construction_pipeline.png)

**Dual-channel quantum augmentation:** Per-stock features and cross-asset correlation structure are encoded through separate quantum circuits, then concatenated for the downstream Ridge model.

![Dual-Channel Architecture](plots/illustrations/real/3_dual_channel_quantum_architecture.png)

**Results** (10 tickers, daily eval, 337 OOS dates × 5 seeds):

| Method | Features | MSE | IC (Pearson r) |
|--------|----------|-----|----------------|
| **Quantum Z+ZZ 8q** (best IC) | 149 | 0.001777 ± 0.000009 | **0.082 ± 0.010** |
| **Quantum Z 8q** (best MSE) | 65 | **0.001768 ± 0.000006** | 0.070 ± 0.014 |
| Identity (no augmentation) | 41 | 0.001779 ± 0.000011 | 0.037 ± 0.006 |
| RFF-96 (best classical) | 137 | 0.001814 ± 0.000014 | 0.044 ± 0.011 |

Quantum methods achieve **2× the information coefficient** of the best classical baselines with lower MSE variance across seeds.

### Feature-to-Qubit Mapping

With 17 input features and 8 qubits, the modular mapping averages features into qubit bins ($q_j \leftarrow \text{mean}(x_i : i \bmod 8 = j)$), compressing without discarding:

![Modular Mapping](plots/illustrations/real/5_modular_feature_qubit_mapping.png)

### Correlation Ablation

A 2×2 factorial study isolates the contributions of regime features and quantum correlation encoding:

![Ablation](plots/real/ablation_grouped_bars.png)

The quantum augmenter (modular mapping) is the top performer in all 4 ablation cells. Cross-asset features provide marginal value beyond the per-stock quantum features for this 10-stock universe.

## Part III — Quantum Hardware Validation

### Rigetti Ankaa-3 Benchmark

The Pareto-optimal quantum reservoir is validated on real QPU hardware via AWS Braket. The reservoir circuit uses only 4 qubits, so we pack **20 independent 4-qubit circuits onto 80 of Ankaa-3's 82 qubits per Braket task** — a ~20× reduction in API calls (and cost) relative to running each sample as its own task.

![Hardware vs Simulator Fidelity](plots/synthetic_hw/three_way_overlay.png)

**Configuration**: 3-reservoir × 3-layer Z-observable reservoir, seed 42, 1000 shots, Rigetti Ankaa-3 and SV1 state-vector simulator for shot-noise-only reference.

| Method | Features | Ridge MSE (n=1000+500) | Δ vs raw |
|---|---:|---:|---:|
| Raw features only (floor) | 4 | 5.66 | — |
| Exact simulator (ceiling) | 16 | **2.59** | **−54 %** |
| Rigetti Ankaa-3 (1000 shots) | 16 | 5.43 | −4 % |

The ~50 % MSE reduction that exact simulation delivers collapses to ≤ 6 % on real hardware across all tested data sizes (100+50, 500+250, 1000+500).

### Noise Characterisation

Fitted two-parameter noise model `R = λ · E + shot_noise(N=1000) + 𝒩(0, σ_g²)`:

![Noise Model Fit Diagnostic](plots/synthetic_hw/noise_model_fit.png)

| Source | λ (damping) | σ_g (gate) | σ_shot | observed corr |
|---|---:|---:|---:|---:|
| SV1 (shot-only self-check) | 1.001 | 0.000 | 0.030 | 0.994 |
| Rigetti singleton 4q | 0.207 | 0.150 | 0.032 | 0.322 |
| Rigetti **packed 80q** (pooled) | **0.258** | **0.159** | 0.032 | 0.337 |

Rigetti Ankaa-3 attenuates Pauli expectations to ~25 % of their ideal magnitude (depolarisation-dominated), with Gaussian gate/readout noise ~5× the shot-noise floor. Qubit-packing density (4q vs 80q active per task) does not produce a measurable incremental crosstalk penalty.

### Can Classical Post-Processing Recover the Signal?

We tested five mitigation strategies — feature-only regression, damping-inversion rescaling by `1/λ`, and top-k feature selection ranked by train-set correlation. **None meaningfully close the gap** between hardware and the exact-simulator ceiling. The single stronger-than-Ridge-CV result (damping correction at n=100+50: 5.46 → 4.64) fails to replicate at larger n and appears to be a finite-α-grid artefact.

**Conclusion**: The information destroyed by hardware noise is not classically recoverable from the saved feature matrix. Closing the gap requires quantum-level error mitigation (ZNE, PEC, dynamical decoupling) or improved device fidelity. As a one-number equivalent, Rigetti's effect on downstream Ridge MSE matches injecting Gaussian noise of **σ ≈ 0.5** into the exact-simulator features.

Full results, tables, and per-figure analysis: [docs/results/synthetic_hw.md](docs/results/synthetic_hw.md). Methodology and caveats: [docs/designs/synthetic_hw.md](docs/designs/synthetic_hw.md). Data-file inventory: [docs/designs/synthetic_hw_data.md](docs/designs/synthetic_hw_data.md).

## Key Conclusions

- **Method**: Pareto-optimal quantum feature augmentation systematically explores 7 structural design dimensions with model-agnostic parameters — isolating encoder **structure** from model choice. On synthetic benchmarks, quantum Pareto outperforms RFF, polynomial, and identity baselines in both linear and nonlinear regimes.

- **Hardware feasibility**: Packing 20 reservoir circuits onto Rigetti Ankaa-3 cuts cost ~20× ($29 vs $540 at n=100+50) with **no measurable density-crosstalk**. HW noise is well-modeled as exact-sim + Gaussian $\sigma \approx 0.5$, matching simulated noise injection.

- **Hardware ceiling**: Only ~15% of Rigetti's error budget is shot noise — 85% is gate + readout + crosstalk, so per-feature fidelity stays at 0.1–0.5 regardless of dataset size. Real-device runs show **significant performance degradation** versus noiseless simulation.

- **Real-data result**: Quantum methods **Pareto-dominate** classical baselines across the full feature-size regime on S&P 500 excess returns; RFF correlation regresses to near-identity levels on stock data. Cross-asset correlation features only meaningfully help naive classical methods; regime scalars carry most of the cross-sectional signal.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/). DataBento API key required for real data experiments.

```bash
uv sync
```

DataBento API key (for real data only):
```bash
echo "DATABENTO_API_KEY=db-..." > .env.databento
```

## Running Experiments

```bash
# ── Synthetic (Part I) ──
uv run python scripts/run_synthetic.py unified     # 1,620 quantum configs factorial sweep
uv run python scripts/run_synthetic.py static      # classical + legacy quantum
uv run python scripts/plot_pareto_vs_classical.py  # Pareto frontier plots

# ── Real data (Part II) ──
uv run python scripts/run_real.py quick            # 3 tickers, monthly, fast validation
uv run python scripts/run_real.py monthly          # 10 tickers, monthly, 5 seeds
uv run python scripts/run_real.py full             # 10 tickers, daily, 5 seeds
uv run python scripts/run_real.py ablation-full    # 2×2 correlation ablation, 5 seeds
uv run python scripts/plot_real_pareto.py          # Pareto + bar plots (5-seed aggregated)
uv run python scripts/plot_real_ablation.py        # Ablation grouped bars + delta plots

# ── Hardware validation (Part III, AWS Braket) ──
uv run python scripts/run_synthetic_hw_packed.py   # pack 20 × 4q reservoirs on Ankaa-3 (80 qubits)
uv run python scripts/run_synthetic_hw_singleton.py # one 4q reservoir per task (clean crosstalk test)
uv run python scripts/plot_synthetic_hw.py         # hardware vs sim MSE plots
uv run python scripts/plot_feature_fidelity.py     # per-measurement fidelity and error histograms
uv run python scripts/fit_noise_model.py           # Analysis 1 — fit (lambda, sigma_g) per source
uv run python scripts/mitigation_study.py          # Analysis 2 — Ridge under 10 mitigation strategies
uv run python scripts/noise_injection.py           # Analysis 3 — match sigma* of injected Gaussian
```

## Project Structure

```
src/synthetic/             # Modular experiment framework (Part I)
  augmenters/
    quantum_unified.py     # Unified 7-dimension quantum augmenter
    quantum_fixed.py       # Legacy quantum augmenters (angle, ZZ, IQP, reservoir, QAOA)
    classical.py           # Polynomial, RFF, oracle, log/abs
    neural.py              # MLP, autoencoder, learned RFF (trainable)
  models/linear.py         # OLS, RidgeCV, LassoCV, ElasticNetCV
  evaluation/              # Metrics, complexity analysis, fairness checking
  runner.py                # Multiprocessing orchestrator
  config.py                # Frozen dataclasses
  dgp.py                   # Synthetic regime-switching DGP

src/real/                  # Real data pipeline (Part II)
  config.py                # RealDataConfig, BacktestConfig, ExperimentConfig
  data.py                  # DataBento OHLCV, feature engineering, cross-asset features
  backtest.py              # Walk-forward backtesting with cached augmentation
  quantum_unified_real.py  # Generalized quantum reservoir (variable input dimension)

scripts/
  run_synthetic.py         # Synthetic experiment entry point
  run_real.py              # Real data entry point (quick/monthly/full/ablation)
  run_synthetic_hw.py      # Hardware experiment entry point (AWS Braket, PennyLane path)
  run_synthetic_hw_packed.py      # Native-Braket packed 20×4q reservoirs (Part III)
  run_synthetic_hw_singleton.py   # pack_factor=1 runbook for clean crosstalk test
  plot_pareto_vs_classical.py  # Synthetic Pareto plots
  plot_real_pareto.py      # Real data Pareto plots (5-seed aggregated)
  plot_real_ablation.py    # Correlation ablation grouped bar plots
  plot_synthetic_hw.py     # Hardware vs simulator MSE plots
  plot_feature_fidelity.py # Per-measurement fidelity and error histograms
  fit_noise_model.py       # Analysis 1 — (λ, σ_g) per source
  mitigation_study.py      # Analysis 2 — 10 mitigation strategies
  noise_injection.py       # Analysis 3 — matched Gaussian σ*
  compare_features.py      # Summary CSV + pairwise feature comparison helper
  quantum_reservoir.py     # Standalone quantum reservoir (library + CLI)

data/                      # Cached datasets (git LFS)
  synthetic/               # Synthetic DGP parquets (10k/10k canonical + small sizes for Part III)
results/                   # Experiment results (JSON + CSV + parquet)
  synthetic_hw/            # Hardware-validation CSVs (summary, noise model, mitigation, noise injection)
plots/                     # Generated figures (HTML + PNG, git LFS)
  synthetic_hw/            # Hardware-validation plots (fidelity, noise model, mitigation, injection)
features/                  # Saved augmented feature matrices (git LFS)
  synthetic_hw/            # Per-run `.npz` for exact / SV1 / Rigetti packed / singleton
docs/                      # Design, results, presentation, changelog
```

## Documentation

See [docs/INDEX.md](docs/INDEX.md) for full documentation index:
- [Challenge Specification](docs/specs/challenge.md)
- [Unified Quantum Design](docs/designs/unified_quantum_design.md) — Factorial design strategy
- [Quantum Reservoir](docs/designs/quantum_reservoir.md) — Optimal circuit design and usage
- [Classical Baselines](docs/designs/classical_baselines.md) — Classical methods and comparison
- [Real Financial Data Design](docs/designs/real.md) — DataBento pipeline, walk-forward backtesting, cross-asset extensions
- [Synthetic-Data Hardware Benchmark — Design](docs/designs/synthetic_hw.md) — Rigetti Ankaa-3 packing / singleton methodology, circuit-family confounders, figure index
- [Synthetic-Data Hardware Benchmark — Data Inventory](docs/designs/synthetic_hw_data.md) — Authoritative index of every saved `.npz` in `features/synthetic_hw/`, filename convention, and variant glossary
- [Synthetic-Data Hardware Benchmark — Results](docs/results/synthetic_hw.md) — Numerical tables, noise-model calibration, mitigation study, matched noise injection

## Challenge Reference

See [AWS-State-Street-Challenge/README.md](AWS-State-Street-Challenge/README.md) for the original challenge specification.
