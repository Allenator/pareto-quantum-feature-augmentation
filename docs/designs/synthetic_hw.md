# Synthetic-Data Hardware Benchmark (Rigetti Ankaa-3)

Documents the figures produced by `scripts/plot_synthetic_hw.py` and
`scripts/plot_feature_fidelity.py`, which benchmark the quantum reservoir
augmentation on the synthetic regime-switching DGP across four sources:

| Source | Backend | Noise |
|---|---|---|
| Exact simulator | `lightning.qubit` | none (analytic) |
| AWS Braket SV1 | SV1 state-vector simulator | shot noise only (1000 shots) |
| Rigetti **singleton** (4q density) | Ankaa-3 QPU, one 4-qubit circuit per task | shot + gate + readout, **no packing crosstalk** |
| Rigetti **packed** (80q density) | Ankaa-3 QPU, 20×4q slots per task | shot + gate + readout + packing crosstalk |

Full data-file inventory and naming convention: see
[synthetic_hw_data.md](synthetic_hw_data.md). All runs share `seed = 42` and
use the same sampled DGP subset — original-feature columns match byte-for-byte
across every pair.

## Configuration caveat — the two circuit families

Two separate Python implementations of the reservoir circuit were used over
the course of the project. Their gate sequences differ, so feature values on
the exact simulator already diverge by up to ±1.5 between the two:

| Family | Source file | Layers structure | Observable set | n_aug |
|---|---|---|---:|---:|
| **A — ReservoirAugmenter** | `src/synthetic/augmenters/quantum_fixed.py` | `n_layers × (Rot + CNOT)`, **no** final Rot | Z+ZZ | 30 |
| **B — QuantumReservoir** | `scripts/quantum_reservoir.py` | `n_layers × (Rot + CNOT)` + **final Rot** | Z only | 12 |

Which family produced which saved file:

| File pattern | Family |
|---|---|
| `reservoir_3x3_Z+ZZ_*.npz` | **A** (ReservoirAugmenter) |
| `reservoir_3x3_Z_*.npz` | **B** (QuantumReservoir) |
| `reservoir_5x3_XYZ+ZZ_*.npz` (SV1) | **A** (ReservoirAugmenter) |

Consequences for analysis:
- Any comparison within a single family is clean (circuit identical, only
  noise source differs).
- Any comparison across families additionally confounds the hardware-noise
  signal with the circuit-family difference. We flag this wherever it
  occurs.

The singleton Rigetti run is Family **A** (Z+ZZ); the packed Rigetti runs are
Family **B** (Z). Thus the two hardware routes cannot be compared on
identical circuits — see *three-way overlay* below for the compromise.

---

## Training-result figures (brief)

### `plots/synthetic_hw/mse_vs_features.png`
Ridge test MSE as a function of augmented-feature count for the three Family
**B** (Z) data sizes. Each of Raw, Exact simulator and Rigetti Ankaa-3 (packed)
has one point per size; marker shape encodes `n_train`. Dashed line at MSE = 1
is the irreducible DGP noise floor.

**Takeaway.** Going from 4 raw features to 16 features with exact-sim reservoir
halves the MSE. With packed Rigetti features at the same dimensionality,
the points sit almost on the raw cluster — hardware noise mostly erases the
augmentation benefit.

### `plots/synthetic_hw/mse_vs_training_size.png`
Same Family **B** runs, x-axis reindexed to training-set size.
Three roughly parallel curves with the exact-sim curve offset below;
the gap between packed Rigetti and raw does not shrink with more samples.
Shot averaging alone does not recover feature quality.

> These two figures are intentionally brief: with only the Family B runs,
> any MSE claim is anecdotal (single seed, small subset). Feature-level
> fidelity (below) is the primary scientific readout from the hardware data.

---

## Feature-fidelity figures (primary focus)

All fidelity plots use **augmented columns only**. The first four columns of
each `.npz` (StandardScaled raw regressors) are identical across sim and
hardware by construction.

Summary numbers (pooled over all augmented features × all samples):

| Source | Family | Observable | n | corr | mean \|err\| |
|---|---|---|---:|---:|---:|
| SV1 shot-only | A | XYZ+ZZ | 100+50 | 0.994 | 0.024 |
| Singleton 4q Rigetti | A | Z+ZZ | 100+50 | 0.322 | 0.200 |
| Packed 80q Rigetti | B | Z | 100+50 | 0.386 | 0.210 |
| Packed 80q Rigetti | B | Z | 500+250 | 0.351 | 0.215 |
| Packed 80q Rigetti | B | Z | 1000+500 | 0.325 | 0.217 |

### `singleton_vs_sim_scatter.png`
**Family A · n=100+50 Z+ZZ · 4q active density · 4 500 pooled measurements.**

Each point plots an (exact, singleton-hardware) pair for one augmented
feature at one sample. Identity line `y = x` is perfect agreement.

**Takeaway.** The cloud is broadly centred on the identity line but has
substantial dispersion, visible out to ±1.0 worst case. Pooled correlation
0.32 — the 4-qubit-density hardware captures some structure but loses most of
the per-feature resolution that the exact simulator provides.

### `packed_vs_sim_scatter.png`
**Family B · Z observable · 80q active density · 1×3 panel over n = 100+50,
500+250, 1000+500.** Same style as above. Marker opacity scaled by point
count so dense panels stay readable.

**Takeaway.** The dispersion around the identity line is invariant with
dataset size (pooled correlations 0.39 → 0.35 → 0.33). Every per-sample
circuit carries the same gate-error fingerprint, so adding more samples does
not average out the noise — it just populates the same cloud more densely.

### `three_way_overlay.png` — cross-family overlay
**Exact-vs-hardware overlay with both singleton (Family A, Z+ZZ) and packed
(Family B, Z) on shared axes.**

Each point is `(feature_exact, feature_hardware)` evaluated within its own
circuit family's exact simulator. Two clouds of different point counts
(singleton 4500, packed 1800) overlap because both hardware sources target
reasonable expectation values in `[-1, +1]`.

**Important confounders (also annotated on the figure):**
- **Circuit family differs** — Family A (no final Rot) for singleton vs
  Family B (with final Rot) for packed. Exact-sim values therefore differ
  between the two clouds for any given input sample.
- **Observable mix differs** — singleton includes 6 pairwise-ZZ products
  per reservoir (which are gate-amplified and tend to be noisier), packed
  is single-Z only.
- **Size control** — both plotted at n=100+50 on the same DGP subset.
- **Same shot budget** — 1000 shots, Ankaa-3, seed 42.

**What we can conclude.** The two clouds have comparable spread around the
identity line: singleton mean \|err\| 0.200 vs packed 0.210. That is
consistent with either interpretation —
1. Crosstalk from 80-qubit density adds only modestly to noise in this
   regime, i.e. the ~5 % difference in mean error is dominated by other
   factors.
2. The comparison is too confounded (circuit + observable + density) to
   isolate crosstalk.

Either way, **we cannot claim from this overlay that packed execution is
worse than singleton**, which was the concern that prompted the crosstalk
investigation in the first place.

### `error_hist.png` — noise-floor decomposition
Overlaid `|exact − source|` probability-density histograms:

1. **SV1** (green, shot only) — mean 0.024, theoretical floor 0.030.
2. **Singleton 4q Rigetti** (mauve) — mean 0.200.
3. **Packed 80q Rigetti, n100+50** (coral) — mean 0.210.
4. **Packed 80q Rigetti, n1000+500** (brown) — mean 0.217.

The dashed vertical line at 0.030 marks the analytic shot-noise floor.

**Takeaway.** SV1 sits tight on the shot-noise floor — no other noise
sources. The three Rigetti histograms look nearly identical to each other,
with heavy right tails extending past \|err\| = 0.8. In rough budget terms:

- **~0.03** shot noise (~15 % of total Rigetti RMS error).
- **~0.17** gate + readout + crosstalk (the remaining ~85 %).

Packing density (4q vs 80q) does not measurably shift this histogram, and
neither does dataset size.

### `per_feature_corr.png` — which features survive?
Per-feature correlation between exact sim and packed Rigetti (Family B, Z
only — 12 features) plotted across the three data sizes. A perfect feature
would be a flat line at `r = 1`.

**Takeaway.** Most features land between 0 and 0.5; two features
(`res 1 · q0`, `res 2 · q0`) sit consistently above 0.4 even at n = 1000+500.
The three best features preserve the most signal; the three worst are near
zero. Rather than trying to denoise every qubit equally, a feature-selection
mitigation using only the three most reliable features is a natural
follow-up (Analysis 2 below).

---

## Files and scripts

| Script | Produces |
|---|---|
| `scripts/plot_synthetic_hw.py` | `mse_vs_features.{html,png}`, `mse_vs_training_size.{html,png}` |
| `scripts/plot_feature_fidelity.py` | `singleton_vs_sim_scatter.{html,png}`, `packed_vs_sim_scatter.{html,png}`, `three_way_overlay.{html,png}`, `error_hist.{html,png}`, `per_feature_corr.{html,png}` |
| `scripts/compare_features.py` | CSV summaries + on-the-fly Ridge MSE comparison |

All figures share the muted Tableau-10 palette used by the pre-existing
synthetic plots (`plot_pareto_vs_classical.py`, `plot_unified.py`,
`plot_synthetic.py`) so colours map consistently across slide decks.

## Follow-up analyses

Without further QPU access, three analyses remain possible from existing
data (see [synthetic_hw_data.md](synthetic_hw_data.md)):

### (1) Noise-model calibration — `fit_noise_model.py` — ✅ complete

Simple two-parameter model fit per source:
`R = λ · E + shot_noise(n=1000) + gate_noise(σ_g)`

| Source | λ (damping) | σ_g (gate) | r observed | r model |
|---|---:|---:|---:|---:|
| SV1 shot-only (Family A, XYZ+ZZ) | 1.001 | 0.000 | 0.994 | 0.994 |
| Singleton 4q (Family A, Z+ZZ) | **0.207** | **0.150** | 0.322 | 0.326 |
| Packed 80q (Family B, Z, pooled) | **0.258** | **0.159** | 0.337 | 0.376 |

Headline findings:
- **Depolarisation dominates.** Rigetti returns Pauli expectations at only
  ~21–26 % of their ideal magnitude — a ~4× attenuation consistent with
  several noisy two-qubit gates per reservoir circuit.
- **Gate/readout σ_g ≈ 0.15**, roughly 5× the shot-noise floor of 0.030.
  Shot noise is not the limiting factor on Ankaa-3 at 1000 shots.
- **SV1 fit recovers λ = 1.00, σ_g = 0.00** — validates the methodology
  (the fit has the right structure, just two parameters, no tuning).
- **Singleton (λ=0.21) looks marginally more damped than packed (λ=0.26)**.
  Counter-intuitive if we expected packing to add crosstalk, but explainable
  by circuit-family and observable-mix differences documented above —
  pairwise-ZZ measurements decohere faster than single-Z. This is *not*
  evidence that packing is crosstalk-free; it is evidence that the dominant
  noise source in our runs is coherent depolarisation, and that the 80 vs 4
  qubit density does not produce a large incremental damping.

See `plots/synthetic_hw/noise_model_fit.png` for the hardware-vs-model error
histogram overlay. Calibration parameters and validation metrics are written
to `results/synthetic_hw/noise_model_fit.csv`.

### (2) Feature-selection mitigation — `mitigation_study.py` — ✅ complete

Compared five strategies across four hardware sources, all vs two reference
bounds (raw-only floor, exact-sim ceiling):

| Source | raw_only (floor) | exact_ideal (ceiling) | hw_baseline | hw_damping_corrected | best hw_top_k | hw_quantum_only |
|---|---:|---:|---:|---:|---:|---:|
| singleton_z+zz_n100+50 | 5.81 | 2.20 | 5.51 | 5.73 | 5.62 (k=8) | 31.6 |
| packed_z_n100+50 | 5.81 | 2.84 | 5.46 | **4.64** | 5.56 (k=8) | 35.0 |
| packed_z_n500+250 | 4.13 | 2.00 | 4.07 | 4.15 | 4.08 (k=8) | 30.2 |
| packed_z_n1000+500 | 5.66 | 2.59 | 5.43 | 5.44 | 5.46 (k=8) | 27.3 |

Takeaways:
- **`hw_quantum_only` > 27 everywhere.** Hardware features alone cannot
  predict Y — Ridge was getting essentially all its signal from the raw
  columns in the hw_baseline configuration.
- **`hw_baseline ≈ raw_only`.** The hardware augmented features add at
  most ~5 % improvement over raw alone. Ridge's cross-validated α already
  suppresses them: explicit feature selection (`hw_topk`) therefore provides
  no further gain — Ridge is already doing the same job implicitly.
- **`hw_damping_corrected` helps only at packed_z_n100+50** (4.64 vs 5.46
  baseline, a 0.82 unit improvement). At the other sizes it has no effect
  or is slightly worse. The one improvement is likely a small-sample
  artefact of the RidgeCV α grid interacting with the feature scaling.
- **No strategy closes the gap between hw_baseline and exact_ideal.** The
  information lost to hardware noise is not classically recoverable from
  the existing feature matrix.

Output: `plots/synthetic_hw/mitigation.png`, `results/synthetic_hw/mitigation.csv`.

**Implication for error mitigation**: the gap can only be closed at the
quantum-circuit level (zero-noise extrapolation, probabilistic error
cancellation, Pauli twirling, or dynamical decoupling) — not by classical
post-processing of the output features.

### (3) Matched classical noise injection — `noise_injection.py` — ✅ complete

For each Rigetti source, inject i.i.d. Gaussian noise of variance σ² into
the **exact-simulator quantum features**, vary σ from 0 to 0.8, and report
the σ* that reproduces the observed Rigetti Ridge test MSE.

| Source | n_train | hw MSE | **σ* (noise-equivalent)** |
|---|---:|---:|---:|
| singleton_z+zz_n100+50 | 100 | 5.51 | 0.42 |
| packed_z_n100+50 | 100 | 5.46 | 0.46 |
| packed_z_n500+250 | 500 | 4.07 | 0.59 |
| packed_z_n1000+500 | 1000 | 5.43 | 0.62 |

Takeaways:
- **σ* grows with training size** — at larger n_train, Ridge can better
  suppress noisy features (more data, more reliable signal estimate), so
  it takes *more* injected noise to degrade Ridge to the hardware's MSE
  level. σ* is therefore not a pure hardware characterization but a joint
  description of "hardware + Ridge on this task".
- **σ* ≈ 0.4–0.6 across configurations** — in plain English, the Rigetti
  Ankaa-3 features behave like exact-sim features corrupted by Gaussian
  noise of standard deviation ~0.5.
- **Complements Analysis 1, does not replace it.** Analysis 1's λ = 0.26,
  σ_g = 0.16 is a per-measurement physical decomposition; Analysis 3's
  σ* = 0.5 is the task-level one-number summary. Both are useful for
  different audiences.

Output: `plots/synthetic_hw/noise_injection.png`,
`results/synthetic_hw/noise_injection.csv`,
`results/synthetic_hw/noise_injection_curve.csv`.

## Synthesis of analyses

Putting Analyses 1, 2, and 3 together:

1. The Rigetti Ankaa-3 output on our reservoir circuit is ~80 % attenuated
   and noisy: λ ≈ 0.25, σ_g ≈ 0.15, σ_shot ≈ 0.03. Packing (4 qubit vs 80
   qubit density) does not materially change this. Analysis 1 provides a
   two-parameter calibrated noise simulator that reproduces the histograms
   and per-feature correlations.
2. Classical post-processing (feature selection, damping inversion,
   quantum-only Ridge) cannot recover signal lost to this noise. The best
   achievable MSE with existing hardware features is indistinguishable
   from raw-only Ridge. Meaningful improvements require quantum-level
   error mitigation.
3. Reported Rigetti MSE on this task is equivalent to running Ridge on
   exact simulator features corrupted by Gaussian noise of σ ≈ 0.5.
