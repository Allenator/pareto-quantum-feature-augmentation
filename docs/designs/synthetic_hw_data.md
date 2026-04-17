# Synthetic-Data Hardware Experiment — Data Inventory

Authoritative listing of every saved feature file used in the synthetic
regime-switching hardware benchmark. Files live under
`features/synthetic_hw/` and are grouped by **source** (`lightning`, `sv1`,
`rigetti`). All runs share `seed = 42` and the same DGP subset, so the first
four (raw) columns match byte-for-byte across every pair.

## Filename convention

```
reservoir_{n_res}x{n_layers}_{observables}_n{n_train}+{n_test}[_s{shots}]_{variant}.npz
```

Fields:
- `n_res × n_layers`: reservoir ensemble size and circuit depth.
- `observables`: which Pauli strings are measured — `Z` (single-qubit only),
  `Z+ZZ` (single + pairwise), `XYZ+ZZ` (3-basis + pairwise).
- `n_train`, `n_test`: sample counts drawn from the same cached DGP parquet.
- `shots`: sampling budget per circuit task. Omitted for analytic (`exact`) runs.
- `variant`: how the run was produced — see table below.

Every `.npz` has two arrays, `train` and `test`, each with shape
`(n_samples, 4 + n_aug)`. Columns 0-3 are the StandardScaled raw regressors;
columns 4+ are the reservoir's augmented features.

## Variant glossary

| Variant | Where it runs | Noise model |
|---|---|---|
| `exact` | `lightning.qubit` local sim | none (analytic) |
| `s1000_sv1` | AWS Braket SV1 state-vector simulator | shot noise only (~σ = 0.03 at 1000 shots) |
| `s1000_packed` | Rigetti Ankaa-3 QPU, 20×4q slots packed on ~80 of 82 qubits per task | shot + gate + readout + inter-slot crosstalk |
| `s1000_singleton` | Rigetti Ankaa-3 QPU, one 4-qubit circuit per task (no packing) | shot + gate + readout, 4-qubit active density |
| `s1000_pennylane_broken` | Rigetti Ankaa-3 via PennyLane parameter broadcasting | **do not use** — plugin artefact, per-group observables collapse to a single value |

## File index

### Exact simulator (`lightning`)

| File | n_train | n_test | Observable | n_aug | Matches |
|---|---:|---:|---|---:|---|
| `reservoir_3x3_Z_n100+50_exact.npz` | 100 | 50 | Z | 12 | packed Rigetti Z n100+50 |
| `reservoir_3x3_Z_n500+250_exact.npz` | 500 | 250 | Z | 12 | packed Rigetti Z n500+250 |
| `reservoir_3x3_Z_n1000+500_exact.npz` | 1000 | 500 | Z | 12 | packed Rigetti Z n1000+500 |
| `reservoir_3x3_Z+ZZ_n5+5_exact.npz` | 5 | 5 | Z+ZZ | 30 | packed Rigetti Z+ZZ n5+5 |
| `reservoir_3x3_Z+ZZ_n100+50_exact.npz` | 100 | 50 | Z+ZZ | 30 | singleton Rigetti Z+ZZ n100+50 |
| `reservoir_5x3_XYZ+ZZ_n100+50_exact.npz` | 100 | 50 | XYZ+ZZ | 90 | SV1 XYZ+ZZ n100+50 |

### AWS Braket SV1 (`sv1`)

| File | n_train | n_test | Observable | n_aug | Shots |
|---|---:|---:|---|---:|---:|
| `reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz` | 100 | 50 | XYZ+ZZ | 90 | 1000 |

Empirical shot-noise RMS on this run is 0.030, matching the theoretical
√((1 − ⟨E²⟩)/N) = 0.0303 for N = 1000 to within 1 %. Fitted shot count via
`(1 − ⟨E²⟩)/⟨err²⟩` = 997. Used as the **shot-noise-only anchor** in the
noise-model calibration.

### Rigetti Ankaa-3 QPU (`rigetti`)

| File | n_train | n_test | Observable | n_aug | Variant |
|---|---:|---:|---|---:|---|
| `reservoir_3x3_Z_n100+50_s1000_packed.npz` | 100 | 50 | Z | 12 | packed |
| `reservoir_3x3_Z_n500+250_s1000_packed.npz` | 500 | 250 | Z | 12 | packed |
| `reservoir_3x3_Z_n1000+500_s1000_packed.npz` | 1000 | 500 | Z | 12 | packed |
| `reservoir_3x3_Z+ZZ_n5+5_s1000_packed.npz` | 5 | 5 | Z+ZZ | 30 | packed |
| `reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz` | 100 | 50 | Z+ZZ | 30 | **singleton (4q-density)** |
| `reservoir_3x3_Z+ZZ_n5+5_s1000_pennylane_broken.npz` | 5 | 5 | Z+ZZ | 30 | broken broadcast — kept only for forensic reference |

The **singleton** run is the earliest Rigetti experiment. It was produced by
`run_synthetic_hw.py` before parameter broadcasting was added: 150 separate
API calls, each running one 4-qubit reservoir circuit on one sample. It is
the only hardware data we have at **4-qubit active density**; all other
Rigetti files run at 80-qubit density (packed).

## How the runs relate

Matching pairs for apples-to-apples feature comparison (same observable, same
data size):

| Exact | Hardware | Observable | n | Purpose |
|---|---|---|---:|---|
| `lightning/…_Z_n100+50_exact.npz` | `rigetti/…_Z_n100+50_s1000_packed.npz` | Z | 100+50 | hardware baseline small |
| `lightning/…_Z_n500+250_exact.npz` | `rigetti/…_Z_n500+250_s1000_packed.npz` | Z | 500+250 | hardware baseline mid |
| `lightning/…_Z_n1000+500_exact.npz` | `rigetti/…_Z_n1000+500_s1000_packed.npz` | Z | 1000+500 | hardware baseline large |
| `lightning/…_Z+ZZ_n5+5_exact.npz` | `rigetti/…_Z+ZZ_n5+5_s1000_packed.npz` | Z+ZZ | 5+5 | tiny-set packed probe |
| `lightning/…_Z+ZZ_n100+50_exact.npz` | `rigetti/…_Z+ZZ_n100+50_s1000_singleton.npz` | Z+ZZ | 100+50 | **4q-density hardware baseline** |
| `lightning/…_XYZ+ZZ_n100+50_exact.npz` | `sv1/…_XYZ+ZZ_n100+50_s1000_sv1.npz` | XYZ+ZZ | 100+50 | **shot-noise-only calibration anchor** |

No packed-vs-singleton pair exists at the same observable and size — the Z
packed runs and the Z+ZZ singleton run differ in the measurement basis. The
singleton's contribution is therefore a qualitative hardware-fidelity
datapoint at low qubit density, not a controlled crosstalk test.

## Scripts that produce these files

| Script | Produces |
|---|---|
| `scripts/run_synthetic_local.py` | `lightning/…_exact.npz` |
| `scripts/run_synthetic_hw.py` (legacy, per-sample path) | `rigetti/…_singleton.npz` (early runs) |
| `scripts/run_synthetic_hw.py` (broadcasting path) | `rigetti/…_pennylane_broken.npz` (avoid) |
| `scripts/run_synthetic_hw_packed.py` | `rigetti/…_packed.npz` |
| `scripts/run_synthetic_hw_singleton.py` | future `rigetti/…_singleton.npz` runs (hardware access required) |
| `scripts/run_synthetic_tn1_packed.py` | `tn1/…` (no runs executed yet) |
| (historical, SV1 run) | `sv1/reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz` |

## Verified input alignment

Every (exact, hardware) pair above has been verified to share the same input
subset: `np.allclose(pair_A["train"][:, :4], pair_B["train"][:, :4], atol=1e-10)`
returns `True`. This is guaranteed by the use of `DGPConfig(seed=42)` with
matching `n_train` / `n_test` across all scripts.
