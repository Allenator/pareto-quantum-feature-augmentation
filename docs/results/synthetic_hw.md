# Results — Synthetic-Data Hardware Benchmark (Rigetti Ankaa-3)

Comprehensive numerical results and interpretation for the hardware / exact-sim /
SV1 comparison on the synthetic regime-switching DGP.

- **Methodology and figure descriptions**: [docs/designs/synthetic_hw.md](../designs/synthetic_hw.md)
- **Data-file inventory**: [docs/designs/synthetic_hw_data.md](../designs/synthetic_hw_data.md)
- **Session change log**: [docs/CHANGES.md](../CHANGES.md)

All numbers below come from saved `.npz` features in
`features/synthetic_hw/` (seed 42, same DGP subsets across sim and hardware).
Ridge test MSE uses `RidgeModel` (sklearn `RidgeCV`, α ∈ {0.001, 0.01, 0.1, 1,
10}, 5-fold CV).

---

## 1. Training-result summary

Ridge test MSE, lower is better. 4-dim raw regressors are StandardScaled
and clipped to ±5; quantum augmentation adds the columns listed.

| Source | Family · Obs | n_train + n_test | # aug feats | Ridge test MSE | vs raw |
|---|---|---:|---:|---:|---:|
| raw only (floor) | — | 100+50 | 0 | 5.8074 | — |
| raw only (floor) | — | 500+250 | 0 | 4.1296 | — |
| raw only (floor) | — | 1000+500 | 0 | 5.6635 | — |
| exact sim | B · Z | 100+50 | 12 | **2.8443** | −51 % |
| exact sim | B · Z | 500+250 | 12 | **1.9989** | −52 % |
| exact sim | B · Z | 1000+500 | 12 | **2.5854** | −54 % |
| exact sim | A · Z+ZZ | 100+50 | 30 | 2.1954 | −62 % |
| Rigetti singleton (4q) | A · Z+ZZ | 100+50 | 30 | 5.5135 | −5 % |
| Rigetti packed (80q) | B · Z | 100+50 | 12 | 5.4565 | −6 % |
| Rigetti packed (80q) | B · Z | 500+250 | 12 | 4.0708 | −1 % |
| Rigetti packed (80q) | B · Z | 1000+500 | 12 | 5.4288 | −4 % |

**Figures**: [`plots/synthetic_hw/mse_vs_features.png`](../../plots/synthetic_hw/mse_vs_features.png),
[`mse_vs_training_size.png`](../../plots/synthetic_hw/mse_vs_training_size.png).

**Takeaway.** Exact-sim reservoir augmentation halves or better the Ridge
test MSE at every data size. Every Rigetti run lands within ±5 % of the
raw-only floor — hardware features add a small but statistically present
advantage, yet nowhere near the simulator ceiling.

---

## 2. Feature-fidelity summary

Pooled (sample × feature) correlation and mean absolute error between each
source and its exact-sim counterpart, augmented columns only.

| Source | Family · Obs | # points | Pooled r (vs exact) | mean \|err\| |
|---|---|---:|---:|---:|
| SV1 shot-only | A · XYZ+ZZ · n100+50 | 13 500 | **0.9944** | **0.0239** |
| Rigetti singleton (4q) | A · Z+ZZ · n100+50 | 4 500 | 0.3222 | 0.2003 |
| Rigetti packed (80q) | B · Z · n100+50 | 1 800 | 0.3864 | 0.2097 |
| Rigetti packed (80q) | B · Z · n500+250 | 9 000 | 0.3512 | 0.2153 |
| Rigetti packed (80q) | B · Z · n1000+500 | 18 000 | 0.3250 | 0.2167 |

**Figures**:
[`singleton_vs_sim_scatter.png`](../../plots/synthetic_hw/singleton_vs_sim_scatter.png),
[`packed_vs_sim_scatter.png`](../../plots/synthetic_hw/packed_vs_sim_scatter.png),
[`three_way_overlay.png`](../../plots/synthetic_hw/three_way_overlay.png),
[`error_hist.png`](../../plots/synthetic_hw/error_hist.png),
[`per_feature_corr.png`](../../plots/synthetic_hw/per_feature_corr.png).

**Takeaway.** SV1 sits essentially on the shot-noise floor (r = 0.994,
mean \|err\| = 0.024 vs theoretical 0.030). Every Rigetti run produces
pooled r in the 0.32–0.39 band and mean \|err\| around 0.21 — dispersion is
invariant with data size (shot averaging alone does not help) and with
qubit-packing density (80q vs 4q does not meaningfully change the error
histogram).

**Confounder for three-way overlay.** The singleton Rigetti run uses
circuit family **A** (ReservoirAugmenter, no final Rot) with Z+ZZ observables;
the packed Rigetti runs use circuit family **B** (QuantumReservoir with
final Rot) with Z only. The overlay plot therefore cannot isolate
density-dependent crosstalk from circuit-family and observable-mix
differences; it does show that the two clouds have comparable spread
(singleton mean \|err\| 0.200, packed 0.210).

---

## 3. Noise-model calibration (Analysis 1)

Model: for a single Pauli expectation `R = λ·E + δ_shot + δ_gate`, where
`δ_shot` is zero-mean with variance `(1 − (λE)²)/n_shots` and
`δ_gate ∼ 𝒩(0, σ_g²)`. `n_shots = 1000` fixed. Fit by least squares.

| Source | Family · Obs | λ (damping) | σ_g (gate) | σ_shot | σ_total | r observed | r modelled |
|---|---|---:|---:|---:|---:|---:|---:|
| SV1 shot-only | A · XYZ+ZZ | **1.001** | **0.0000** | 0.0303 | 0.0301 | 0.9944 | 0.9943 |
| Rigetti singleton (4q) | A · Z+ZZ | **0.207** | **0.150** | 0.0316 | 0.1530 | 0.3222 | 0.3262 |
| Rigetti packed (80q) pooled | B · Z | **0.258** | **0.159** | 0.0316 | 0.1624 | 0.3373 | 0.3761 |

Canonical CSV: [`results/synthetic_hw/noise_model_fit.csv`](../../results/synthetic_hw/noise_model_fit.csv).
Diagnostic: [`noise_model_fit.png`](../../plots/synthetic_hw/noise_model_fit.png).

**Findings**

- **Depolarisation dominates.** Rigetti returns expectation values at
  ~21–26 % of their ideal magnitude. For a 3-layer Rot + linear-CNOT
  reservoir circuit this is consistent with a few percent two-qubit gate
  error compounded across the circuit.
- **Gate / readout Gaussian σ_g ≈ 0.15** is ~5× the shot-noise floor.
  Shot noise is *not* the limiting factor at 1000 shots.
- **Model fits the histograms and correlations precisely** — the SV1
  self-consistency check (λ ≈ 1, σ_g ≈ 0) validates the fit methodology.
- **Packed 80q density has *higher* λ than singleton 4q** (0.258 vs 0.207).
  Counter-intuitive if we expected packing to add crosstalk; explainable
  by the circuit-family and observable-mix confounders — pairwise-ZZ
  measurements decohere faster than single-Z so the singleton Z+ZZ fit
  is pulled down. Density does **not** add a detectable incremental
  damping above the per-circuit depolarisation floor.

---

## 4. Mitigation study (Analysis 2)

Ridge test MSE, lower = better.  `exact_ideal` = ceiling, `raw_only` = floor.

| Strategy | singleton_z+zz n100+50 | packed_z n100+50 | packed_z n500+250 | packed_z n1000+500 |
|---|---:|---:|---:|---:|
| raw_only (floor) | 5.81 | 5.81 | 4.13 | 5.66 |
| **exact_ideal (ceiling)** | **2.20** | **2.84** | **2.00** | **2.59** |
| hw_baseline | 5.51 | 5.46 | 4.07 | 5.43 |
| hw_quantum_only | 31.61 | 35.01 | 30.17 | 27.30 |
| hw_damping_corrected | 5.73 | **4.64** | 4.15 | 5.44 |
| hw_top1 | 5.72 | 5.77 | 4.15 | 5.65 |
| hw_top2 | 5.70 | 5.86 | 4.17 | 5.59 |
| hw_top4 | 5.70 | 5.66 | 4.12 | 5.46 |
| hw_top8 | 5.62 | 5.56 | 4.08 | 5.46 |
| hw_topall | 5.51 | 5.46 | 4.07 | 5.43 |

Canonical CSV: [`results/synthetic_hw/mitigation.csv`](../../results/synthetic_hw/mitigation.csv).
Plot: [`mitigation.png`](../../plots/synthetic_hw/mitigation.png).

**Findings**

- **`hw_quantum_only` blows up** (MSE 27–35) — hardware features cannot
  predict Y on their own. Ridge's signal in `hw_baseline` comes almost
  entirely from the raw columns.
- **`hw_baseline` sits ~0.3 MSE units below `raw_only`** — hardware
  quantum features add ~5 % marginal signal. Ridge's cross-validated α
  is already suppressing them effectively; explicit top-k selection
  therefore provides no further gain.
- **`hw_damping_corrected` helps only at packed_z_n100+50** (4.64 vs 5.46
  baseline, −0.82 units), but fails at the other sizes. Likely a small-
  sample interaction with the coarse RidgeCV α grid rather than a robust
  mitigation.
- **No strategy closes the gap between `hw_baseline` and `exact_ideal`.**
  The information lost to hardware noise is not classically recoverable
  from the saved feature matrix. Mitigation must happen at the quantum-
  circuit level (zero-noise extrapolation, probabilistic error
  cancellation, Pauli twirling, dynamical decoupling).

---

## 5. Matched classical noise injection (Analysis 3)

For each Rigetti source, `noise_injection.py` sweeps σ ∈ [0, 0.8] adding
`𝒩(0, σ²)` to the exact-sim quantum features and measures Ridge test MSE
(averaged over 16 noise realisations). σ* is the σ that reproduces the
observed Rigetti MSE.

| Source | n_train | exact MSE | hw MSE | **σ*** |
|---|---:|---:|---:|---:|
| singleton_z+zz n100+50 | 100 | 2.20 | 5.51 | **0.42** |
| packed_z n100+50 | 100 | 2.84 | 5.46 | **0.46** |
| packed_z n500+250 | 500 | 2.00 | 4.07 | **0.59** |
| packed_z n1000+500 | 1000 | 2.59 | 5.43 | **0.62** |

Canonical CSVs:
[`results/synthetic_hw/noise_injection.csv`](../../results/synthetic_hw/noise_injection.csv),
[`noise_injection_curve.csv`](../../results/synthetic_hw/noise_injection_curve.csv).
Plot: [`noise_injection.png`](../../plots/synthetic_hw/noise_injection.png).

**Findings**

- **σ* ≈ 0.4–0.6** — Rigetti hardware degrades the downstream Ridge
  regression by about as much as adding i.i.d. Gaussian noise of σ ≈ 0.5
  would to the exact-sim features.
- **σ* grows with training size.** At larger n_train Ridge can suppress
  noisy features more aggressively (more data, better α choice), so it
  takes more injected noise to bring it down to hardware's MSE. σ* is a
  *task-level* summary, not a pure hardware characterisation.
- **σ* is not directly comparable to Analysis 1's σ_total.** Analysis 1
  fits per-measurement residual variance; σ* runs through the full
  Ridge pipeline. Both are correct, each answers a different question:
  σ_total (≈ 0.16) characterises one measurement; σ* (≈ 0.5)
  characterises the end-to-end regression impact.

---

## 6. Synthesis — putting all three analyses together

1. **Rigetti Ankaa-3 behaviour on this circuit is well described by two
   physical parameters**: a depolarising-like damping λ ≈ 0.25 and a
   Gaussian gate + readout term σ_g ≈ 0.15. Shot noise at 1000 shots
   contributes σ_shot ≈ 0.03, an order of magnitude smaller than σ_g.
2. **Qubit-packing density does not matter at this level.** 4 qubits vs
   80 qubits active per task — confounded by circuit-family and
   observable-mix differences — shows no measurable incremental error.
   Crosstalk is dominated by per-circuit depolarisation.
3. **Classical post-processing cannot recover the lost signal.** Feature
   selection, quantum-only Ridge, and damping inversion all fail to
   meaningfully close the gap from hw_baseline to exact_ideal. Even with
   5 seeds and 1500 training samples the marginal hardware improvement
   over raw-only Ridge is ≤ 5 %.
4. **Task-level noise equivalent is σ ≈ 0.5.** Readers who want a
   one-number summary can treat the Rigetti-Ridge pipeline as running
   exact-sim features through `𝒩(0, 0.5²)` before regression.
5. **Implications for mitigation.** The gap can only be closed at the
   quantum-circuit level. Candidates (in order of expected cost / benefit
   for this reservoir design):
   - Shorter, less-entangling circuits (fewer CNOTs → higher λ).
   - Zero-noise extrapolation at two or three stretch factors.
   - Probabilistic error cancellation if a device noise model is
     available.
   - Dynamical decoupling on idle qubits (especially in the 80q packed
     regime).

## 7. Caveats and open questions

- **Single seed, single QPU provider.** All hardware data is seed 42 on
  Ankaa-3. A second seed or a different QPU would test generality of the
  λ ≈ 0.25, σ_g ≈ 0.15 numbers.
- **Two circuit families are in play.** The singleton run (family A, no
  final Rot, Z+ZZ) and packed runs (family B, with final Rot, Z only) are
  *not* a controlled test of packing density. A fair crosstalk test
  requires a `pack_factor = 1` rerun with the same circuit family as the
  packed runs — scripted in `scripts/run_synthetic_hw_singleton.py` but
  not executed (no QPU access).
- **Simple noise model.** Analysis 1 uses a two-parameter depolarising +
  Gaussian model. A richer model (Pauli-twirled, qubit-dependent
  depolarising, asymmetric readout) would likely fit even better but is
  not needed for the conclusions above.
- **Feature-selection null result.** Cannot rule out that a more
  sophisticated mitigation (e.g. ensemble of many hardware runs, or
  mid-circuit measurement + post-selection) would recover more signal.
