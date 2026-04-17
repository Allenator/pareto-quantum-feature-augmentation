# Changelog — Synthetic-Data Hardware Benchmark

Log of additions and modifications made while analysing the existing Rigetti
Ankaa-3 / SV1 / exact-simulator feature files. Scope: the
`features/synthetic_hw/` subtree and everything that consumes it.

## New scripts

| Path | Purpose |
|---|---|
| `scripts/compare_features.py` | Load any number of `.npz` feature files, auto-add a raw-features-only baseline, print Ridge MSE summary, run pairwise feature-level comparisons, and export three CSVs (summary, pairwise correlations, per-feature detail). |
| `scripts/plot_synthetic_hw.py` | Produce two **single-panel** PNG/HTML figures: `mse_vs_features` and `mse_vs_training_size`. (Split from an earlier two-panel figure.) |
| `scripts/plot_feature_fidelity.py` | Produce five feature-fidelity plots: `singleton_vs_sim_scatter`, `packed_vs_sim_scatter` (1×3 panels), `three_way_overlay`, `error_hist` (with SV1 shot-noise reference), `per_feature_corr`. Replaces the earlier `three_way_scatter` / `three_way_error_hist` which used a broken PennyLane-broadcasting run. |
| `scripts/run_synthetic_hw_singleton.py` | Hardware runbook (not executed — no QPU access) for `pack_factor = 1` Rigetti runs. Reuses `build_packed_circuit` from the packed script. |
| `scripts/fit_noise_model.py` | **Analysis 1**: fit `R = λ·E + shot_noise(n=1000) + 𝒩(0, σ_g)` per source (SV1 shot-only, singleton 4q Rigetti, packed 80q Rigetti pooled). Output CSV + diagnostic overlay histogram. |
| `scripts/mitigation_study.py` | **Analysis 2**: compare Ridge test MSE across 10 strategies — `raw_only`, `exact_ideal`, `hw_baseline`, `hw_quantum_only`, `hw_damping_corrected`, `hw_top{1,2,4,8,all}`. Output CSV + grouped bar chart. |
| `scripts/noise_injection.py` | **Analysis 4**: find the Gaussian σ* such that `exact + 𝒩(0, σ*)` matches each Rigetti Ridge MSE. Output CSV + MSE-vs-σ sweep plot with σ* markers. |

Dependencies are unchanged — all scripts use the existing `numpy`, `plotly`,
`sklearn`, `amazon-braket-sdk` stack plus the in-repo `src/synthetic/`
helpers and `scripts/quantum_reservoir.py`.

## Renamed feature files

Every saved `.npz` under `features/synthetic_hw/` now follows the
convention `{augmenter}_n{n_train}+{n_test}[_s{shots}]_{variant}.npz`:

| Before | After |
|---|---|
| `rigetti/seed_42/reservoir_3x3_Z+ZZ.npz` | `rigetti/seed_42/reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz` |
| `lightning/seed_42/reservoir_5x3_XYZ+ZZ.npz` | `lightning/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_exact.npz` |
| `sv1/seed_42/reservoir_5x3_XYZ+ZZ.npz` | `sv1/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz` (shot count verified empirically, N = 997 ≈ 1000) |
| `rigetti/seed_42/reservoir_3x3_Z+ZZ_n5+5_s1000_pennylane.npz` | `rigetti/seed_42/reservoir_3x3_Z+ZZ_n5+5_s1000_pennylane_broken.npz` (broken PennyLane parameter-broadcasting artefact; renamed so future readers skip it) |

## Deleted files

| Path | Reason |
|---|---|
| `lightning/seed_42/reservoir_3x3_Z+ZZ.npz` | Byte-identical to `reservoir_3x3_Z+ZZ_n100+50_exact.npz` — exact sim of the same config and size. |

## New / updated documentation

| Path | Status | Purpose |
|---|---|---|
| `docs/designs/synthetic_hw.md` | updated | Design doc — methodology, circuit families, confounders, figure index. Analyses 1/2/4 promoted from *planned* to ✅ complete. |
| `docs/designs/synthetic_hw_data.md` | new | Authoritative per-file inventory, filename convention, variant glossary, verified (exact, hardware) input-alignment pairs. |
| `docs/results/synthetic_hw.md` | new | Comprehensive numerical results and analysis. |
| `docs/INDEX.md` | updated | Registers the new docs. |
| `docs/CHANGES.md` | new (this file) | Session-level change log. |

## New plots

All under `plots/synthetic_hw/`, both `.html` and `.png` (scale = 3):

| File | Produced by | What it shows |
|---|---|---|
| `mse_vs_features.png` | `plot_synthetic_hw.py` | Ridge test MSE vs # features across methods and sizes. |
| `mse_vs_training_size.png` | `plot_synthetic_hw.py` | Ridge test MSE vs training-set size. |
| `singleton_vs_sim_scatter.png` | `plot_feature_fidelity.py` | Exact vs Rigetti singleton pooled per-measurement scatter, Z+ZZ n100+50. |
| `packed_vs_sim_scatter.png` | `plot_feature_fidelity.py` | Exact vs Rigetti packed pooled scatter, 1×3 panel across sizes. |
| `three_way_overlay.png` | `plot_feature_fidelity.py` | Singleton + packed hardware-vs-exact overlay, with circuit-family caveat annotated. |
| `error_hist.png` | `plot_feature_fidelity.py` | Error-distribution histogram comparing SV1 shot-noise floor, singleton, packed small, packed large. |
| `per_feature_corr.png` | `plot_feature_fidelity.py` | Per-feature correlation (packed vs exact) across data sizes. |
| `noise_model_fit.png` | `fit_noise_model.py` | Error histograms of real hardware vs calibrated noise-model synthesis. |
| `mitigation.png` | `mitigation_study.py` | Grouped bar chart of Ridge MSE per mitigation strategy and source. |
| `noise_injection.png` | `noise_injection.py` | MSE vs σ sweep with σ* markers. |

## New result CSVs

Under `results/synthetic_hw/`:

| File | Produced by | Content |
|---|---|---|
| `summary.csv` | `compare_features.py` | Per-dataset Ridge MSE, residual stats, and % change vs raw. |
| `prediction_correlations.csv` | `compare_features.py` | Pairwise prediction-correlation matrix. |
| `feature_detail.csv` | `compare_features.py` | Per-feature mean/max \|diff\| and correlation. |
| `noise_model_fit.csv` | `fit_noise_model.py` | (λ, σ_g, σ_shot, σ_total, r_observed, r_model) per source. |
| `mitigation.csv` | `mitigation_study.py` | Ridge test MSE per (source, strategy). |
| `noise_injection.csv` | `noise_injection.py` | Matched σ* per source. |
| `noise_injection_curve.csv` | `noise_injection.py` | Full σ-MSE sweep for each source. |

## Key scientific findings added during this work

Full tables and interpretation: see [docs/results/synthetic_hw.md](results/synthetic_hw.md).
Short version:

1. Rigetti Ankaa-3 output is ~75 % attenuated and ~0.15 noisy: the fitted
   depolarising-like damping λ ≈ 0.25 and Gaussian gate noise σ_g ≈ 0.15
   dominate shot noise (σ_shot ≈ 0.03) by a factor of ~5.
2. 80 qubit vs 4 qubit active density does not materially shift the fit —
   packing does not add measurable crosstalk above the per-circuit
   depolarisation floor on this circuit.
3. No classical post-processing on existing features recovers the
   information lost to hardware noise.
4. The hardware output is classically equivalent to adding Gaussian noise
   of σ ≈ 0.5 to the exact-sim features, before downstream Ridge.
