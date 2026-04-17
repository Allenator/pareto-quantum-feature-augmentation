# Presentation — Preparation Document

Your pre-presentation read. For each slide: the full context, the exact numbers, what can go wrong, and what questions the audience is likely to ask. You should be able to present this section even if the slides break.

Companions: [slides.md](slides.md) (raw slide content) · [outline.md](outline.md) (on-stage quick reference) · [docs/results/synthetic_hw.md](../../results/synthetic_hw.md) (canonical results).

---

## Slide 1 — Available AWS hardware

### Context

AWS Braket is Amazon's quantum-computing-as-a-service offering. It exposes several third-party QPUs through a unified API, plus Amazon-operated simulators (SV1 state-vector, DM1 density-matrix, TN1 tensor-network).

At the time of our project:

| Provider · Device | Type | Available to us? |
|---|---|---|
| IonQ (Aria-1 / Forte) | Trapped-ion | No — provider blocked / queue-restricted |
| D-Wave (Advantage) | Annealer | No — gate-model project |
| IQM Garnet | 20q superconducting | No — Stockholm region access denied |
| **Rigetti Ankaa-3** | **82q superconducting** | **Yes** (us-west-1) |
| **AWS SV1** | State-vector sim, ≤34 qubits | **Yes** (on-demand) |
| AWS TN1 | Tensor-network sim | Available; we didn't run any experiments on it |
| AWS DM1 | Density-matrix sim | Available; not used |

**Ankaa-3 topology**: rectangular lattice, 82 addressable qubits, nearest-neighbour connectivity. The layout is shown in `plots/rigetti_ankaa3.svg` — two-column grid of qubits connected to immediate neighbours. Two-qubit native gate is an entangling gate; our circuits use CNOT (decomposed automatically).

### Likely questions

- **Why not just run more simulator?** Simulation is fine for 4-qubit circuits but the *point* of the project is to evaluate real quantum hardware. If we only simulated, there would be no hardware story.
- **What would you have done with IonQ?** IonQ has all-to-all connectivity and much higher gate fidelity — we'd expect less depolarisation (higher λ, slide 10). But the reservoir circuit is small enough that Ankaa-3 is a reasonable platform for this specific task.

---

## Slide 2 — Cost constraints

### Context

Braket pricing has two components:
- **Per-task fee**: $0.30 for Rigetti Ankaa-3 and IQM Garnet. This is charged once per submitted task, regardless of circuit size.
- **Per-shot fee**: $0.0009 on Rigetti Ankaa-3 (IQM Garnet is $0.00145). One shot = one measurement of the circuit in the computational basis.

SV1 is priced per-minute ($0.075/min), so for our small circuits the cost is negligible.

### Cost estimation for our experiment

Our augmenter is `reservoir_3x3_Z`: 3 random reservoir circuits, 3 mixing layers each, Z-basis measurements. Each reservoir produces 4 features (one ⟨Zᵢ⟩ per qubit) → 12 augmented features total. 4-qubit circuits.

With `pack_factor = 20` (see slide 3), we submit only `ceil(N_samples / 20) × 3` tasks for N_samples total data points across 3 reservoirs:

| n_train + n_test | Packed tasks | Task cost | Shot cost (1000/shot) | Total |
|---:|---:|---:|---:|---:|
| 150 | 24 | $7.20 | $21.60 | **$29** |
| 750 | 114 | $34.20 | $102.60 | **$137** |
| 1500 | 225 | $67.50 | $202.50 | **$270** |

Unpacked (pack_factor = 1) would be roughly 20× more tasks at the same scale — e.g., $540 for n=150 instead of $29. Even at our largest scale, unpacked would cost $5400+.

### Likely questions

- **What's the total we spent?** Sum of the three packed runs is ~$436 for Rigetti, plus one legacy singleton run at 100+50 (~$135). Total around $570 across the whole project.
- **Why didn't you just use SV1 for everything?** SV1 has no hardware noise — it would have shown only shot noise and would not have revealed the gate/readout/decoherence issues that are the main scientific finding.
- **Why 1000 shots specifically?** Rule of thumb: shot noise σ ≈ √((1−E²)/N). At 1000 shots, σ ≈ 0.03 — small relative to typical expectation magnitudes but not so many shots that cost dominates. We verified empirically via SV1 that 1000 shots gives the expected shot-noise floor.

---

## Slide 3 — Packed execution on Ankaa-3

### Context

Each of our reservoir circuits uses only 4 qubits. Ankaa-3 has 82. We exploit this by running 20 *independent* 4-qubit reservoirs on disjoint qubit subsets in a single Braket task.

- Qubits 0–3: reservoir copy for sample 1
- Qubits 4–7: reservoir copy for sample 2
- …
- Qubits 76–79: reservoir copy for sample 20
- (Qubits 80–81 unused.)

Each slot has its own `RY(x_i)` data encoding followed by the random `Rot + CNOT` layers; measurements on each slot are independent expectations. The task returns 20 × 4 = 80 expectation values, one batch of 20 samples at a time.

Crucial detail: within one Braket task, circuits in different slots share:
- One per-task submission fee
- One shot budget (1000 shots → each slot gets 1000 shots because the measurements are disjoint qubits)
- Simultaneous execution on shared hardware — potential source of crosstalk

### Footnote — PennyLane / circuit family

- **Early attempt**: use PennyLane's parameter broadcasting to submit N samples in one Braket task. The Braket plugin collapsed per-observable values within each commuting measurement group (all four ⟨Zᵢ⟩ would come back identical, then all six ⟨ZᵢZⱼ⟩ identical, etc.) — features became pathologically low-rank.
- We discovered this by comparing the n=5+5 pennylane output row-by-row to the exact sim: each reservoir's features had only 2 distinct values instead of 10.
- Fix: native Braket `Circuit` construction with `pack_factor = 20`. Each observable becomes its own `expectation()` spec and returns distinct, well-conditioned values.
- **Time pressure**: the earliest Rigetti runs (before we finalised the better augmenter) used `ReservoirAugmenter` with Z+ZZ observables. Later packed runs use the improved `QuantumReservoir` circuit (with a final Rot layer) and Z-only observables. This is a confounder for the three-way overlay on slide 9.

### Likely questions

- **Why 20 slots and not 10 or 40?** 20 × 4 = 80 fits cleanly into Ankaa-3's 82 qubits with 2 spare. Fewer slots would waste qubits; more slots would overflow.
- **Can parallel slots interfere?** In principle yes — Rigetti's gates aren't perfectly isolated. The three-way overlay on slide 9 tests this empirically; we don't see a measurable effect above per-circuit depolarisation.
- **Why share shots instead of per-slot shots?** Measurements on disjoint qubit subsets are statistically independent in each computational-basis shot. 1000 shots gives each slot 1000 samples of its own subspace.

---

## Slide 4 — Headline results

### Context

Ridge regression with cross-validated α ∈ {0.001, 0.01, 0.1, 1, 10}, 5-fold CV. Features are StandardScaled raw + (optionally) augmented quantum features. Reported metric is test MSE on held-out samples.

### Reading the table

| Row | What it shows |
|---|---|
| Raw only | Floor — what we'd get with no quantum augmentation. MSE 5.8 / 4.1 / 5.7 across three sizes. |
| Noiseless sim | Ceiling — what perfect quantum features deliver. MSE 2.8 / 2.0 / 2.6 → about half the raw MSE. |
| Rigetti 1000 shots | Real hardware. MSE 5.5 / 4.1 / 5.4 → essentially the raw floor, maybe 1–6 % better. |

Columns to note:
- **mse_vs_raw**: absolute MSE change vs the raw baseline for the same n.
- **mse_pct_change_vs_raw**: the quantum augmentation benefit in relative terms. Noiseless sim: **−51 to −54 %**. Rigetti: **−1 to −6 %**.
- **residual μ, σ**: prediction-error distribution. Noiseless sim has near-zero mean (unbiased) and σ ≈ 1.6. Hardware has the same bias and σ as raw — Ridge essentially ignores the noisy quantum features.

### Why n=500 is weird

The raw-only MSE at n=500 is 4.1 instead of ~5.7 like the other two sizes. This is a seed-specific artefact — that particular subset has lighter tail outliers, so raw Ridge does better. The gap between raw and exact-sim ceiling is stable (~52 %) across all sizes; that's the real signal.

### Likely questions

- **Why doesn't Ridge just use the quantum features if they have some signal?** It does, but the features are so attenuated and noisy that α-regularisation shrinks their coefficients toward zero. Slide 11 probes this directly.
- **Is the 6 % improvement at n=100 statistically meaningful?** Single seed, so we can't give a confidence interval. Probably real but noisy — consistent with the ~15 % of features that have r > 0.4 in the per-feature correlation plot.
- **What about 2000 or 5000 samples?** Would cost ~$400–$1000 more. The curve on slide 12 suggests σ* grows roughly logarithmically with n_train, so more data eventually makes Ridge entirely ignore the quantum features. Not worth the spend.

---

## Slide 5 — MSE visualisation placeholder

### Context

Originally `plot_synthetic_hw.py` produced `mse_vs_features.png` (a log-x plot at #features 4 and 16 per method) and `mse_vs_training_size.png` (line chart of MSE vs n_train per method). Both are visually uninformative — at most 6–9 data points, all clustered on two x-values for the features plot, and essentially flat lines for the sizes plot.

### What should replace them

A **grouped bar chart**: x-axis = dataset size (100+50, 500+250, 1000+500), bars grouped by method (Raw / Noiseless sim / Rigetti), y-axis = Ridge test MSE. This makes the headline story visually immediate: the Rigetti bars are pinned to the Raw bars at every size, while the Noiseless sim bars sit at ~half height.

Existing canvas: `plots/synthetic_hw/mitigation.png` uses this exact style. A matching 3-method 3-size version would be about 15 lines of Plotly to add to `plot_synthetic_hw.py`.

### What to say on stage

Skip this slide, or briefly note: "bar chart to come — numbers are in the previous table."

---

## Slide 6 — Per-feature fidelity

### Context

Compute per-feature Pearson correlation between exact simulator and packed Rigetti, for each of the 12 augmented features, at each of the three training sizes. Plot as line chart: one line per feature, x = n_train, y = correlation.

### What to highlight

- All lines are below 1 (the green dashed reference) — perfect hardware would be a flat line at 1.
- Most lines are between 0.1 and 0.5.
- Two features — `res 1 · q0` (green) and `res 2 · q0` (brown) — stay above 0.4 across all three sizes.
- At least one feature (`res 1 · q3`, pink) drops toward zero at larger n.
- The ranking of features by fidelity is fairly stable across sizes — it's a hardware-circuit property, not a small-sample fluke.

### Interpretation

Different qubit positions within a reservoir have different noise characteristics. On Rigetti, qubit-0 seems to consistently do better than qubit-3 across all three reservoirs. Possible causes:
- Qubit-specific T1/T2 coherence times
- Readout fidelity varying by qubit
- The linear-CNOT mixing layer places qubit-0 at the "upstream" end, so it accumulates less CNOT error than qubit-3 at the downstream end.

### Likely questions

- **Could we just use the top-2 features?** Tested on slide 11 (`hw_top2`, `hw_top4`) — didn't help, because Ridge is already effectively filtering out the noisy ones via regularisation.
- **Why does r drop at larger n?** Counter-intuitive — more data usually helps. Our best guess: as the sample-based correlation estimate tightens, it reveals the true (low) correlation more precisely. At small n, the estimate is noisy and occasionally high by chance.

---

## Slide 7 — Per-measurement scatter

### Context

Three-panel plot, one per data size. For each size:
- x-axis: exact-simulator expectation value for one (sample, feature) measurement
- y-axis: corresponding packed-Rigetti expectation value
- Dashed line y=x = perfect agreement
- Opacity scales with point count so big panels stay readable

Pooled correlation coefficients on-panel: 0.39 / 0.35 / 0.33 (slight drop with n, same effect as on slide 6).

### What to highlight

- The cloud **does not shrink** with more data. Denser, yes — tighter, no.
- Roughly symmetric about the identity line — no visible bias. (Detailed analysis on slide 10 shows there *is* a multiplicative bias, but it's hidden in the scatter because the scatter is dominated by noise.)
- Outliers extending to \|err\| > 1 — some measurements are wildly off.

### Interpretation

The noise is **per-circuit**, not per-shot. Each submitted circuit inherits a full gate-error "fingerprint" — imperfect gates, decoherence during execution, readout flips. Running more samples exposes the same noise distribution more densely but doesn't cancel it.

### Likely questions

- **Why does correlation drop with more n?** Same as slide 6 — small-sample correlation is noisier and the true value is low.
- **What would fix this?** Better gates (hardware upgrade), shorter circuits (fewer CNOTs → less decoherence), or quantum error mitigation (ZNE, PEC).

---

## Slide 8 — Error histogram and shot-noise floor

### Context

For each source, compute \|exact − source\| for every (sample, feature) measurement and plot as probability density histogram.

Four traces:
1. **SV1** (green): state-vector simulator with 1000-shot sampling. Only noise source is shot sampling.
2. **Singleton 4q Rigetti** (mauve): 100+50 samples, Z+ZZ observables, 4-qubit-density.
3. **Packed 80q Rigetti n=100+50** (coral): 12 features, Z-only.
4. **Packed 80q Rigetti n=1000+500** (brown): same as above with 10× more data.

Dashed vertical at 0.030 = theoretical shot-noise floor at 1000 shots (σ ≈ √((1−E²)/1000) ≈ 0.030 for small E).

### What to highlight

- SV1 histogram peaks right at 0.030 and has a narrow tail — shot-only behaviour, exactly as expected.
- All three Rigetti histograms **overlap each other almost perfectly** and peak around 0.05–0.10 with tails out past 0.8.
- The Rigetti histograms are separated from SV1 by ~5× in peak location.

### Decomposition

Using mean absolute error as a rough proxy:
- SV1 mean \|err\| ≈ 0.024 (shot noise only)
- Rigetti mean \|err\| ≈ 0.20
- Fraction of Rigetti error from shot noise: ~0.024 / 0.20 ≈ 12 %
- Remaining ~88 % is gate + readout + possible crosstalk.

Another way to see it: the SV1 histogram has essentially zero density past \|err\| = 0.1, while Rigetti has massive density in that regime.

### Likely questions

- **What if we ran Rigetti with 10 000 shots?** Shot-noise floor drops to ~0.010, but gate/readout noise is unchanged. Total error would drop only marginally (from 0.20 to ~0.19). The bottleneck is not shots.
- **Can error mitigation reduce the gate-noise component?** Yes in principle — ZNE extrapolates to zero noise, PEC inverts known errors. Both require more circuit executions per data point, so cost goes up.

---

## Slide 9 — Three-way overlay

### Context

We want to know: does packing 20 independent circuits onto 80 qubits add measurable crosstalk compared to running each on 4 qubits alone?

**Data available:**
- Singleton (4q density): 100+50 samples, Z+ZZ observables (30 features). From the very first Rigetti run, before we added the broadcasting bug and before we finalised the augmenter. Produced by `run_synthetic_hw.py` sample-by-sample.
- Packed (80q density): 100+50 samples, Z-only observables (12 features). From `run_synthetic_hw_packed.py`.

Both hardware sources plotted against **their own** exact-simulator references on shared axes.

### What to highlight

- Two clouds overlap around the identity line. Spread is visually similar.
- Singleton mean \|err\| = 0.200 (4500 points). Packed mean \|err\| = 0.210 (1800 points).
- Singleton r = 0.32. Packed r = 0.39. Packed is actually slightly tighter.

### Caveat (must say out loud)

This is **not** a controlled crosstalk test. Confounders:
1. **Different reservoir circuit**: singleton uses `ReservoirAugmenter` (no final Rot); packed uses `QuantumReservoir` (with final Rot). The reservoir-augmented features are literally different functions of the input.
2. **Different observables**: singleton measures Z+ZZ; packed measures Z only. Pairwise-ZZ products lose coherence faster than single Z, so singleton is *at a disadvantage* on this axis, which cancels any packing advantage singleton might have.
3. Same input subset (same seed, same samples), same shot count — those parts are clean.

### Why it still tells us something

Even with these confounders, we'd expect packing 80 qubits to *dramatically* worsen error if crosstalk were dominant — an extra 0.1 or more on mean \|err\|. We see a 0.010 difference. That's enough to conclude: **crosstalk is not the dominant error on Ankaa-3 for this circuit**.

### Why we couldn't do a clean test

A clean test needs the same circuit family on both routes. We drafted `run_synthetic_hw_singleton.py` (ready to run, commented up) but lost QPU access before we could execute it. Cost estimate for the clean rerun: ~$540 for the 100+50 case.

### Likely questions

- **Why was the earliest run not using the improved augmenter?** Time pressure — we had to run on hardware as soon as QPU access was confirmed, before the augmenter work was finalised.
- **Are you sure 80q doesn't add crosstalk on any circuit?** No — we only tested this reservoir. Other circuits (deeper, more entangling, using long-range couplings) would likely see more.

---

## Slide 10 — Noise-model calibration (Analysis 1)

### Context

We fit a two-parameter phenomenological noise model:

```
R = λ · E + δ_shot(N=1000) + δ_gate(σ_g)
```

- `R`: hardware measurement of a single Pauli expectation
- `E`: exact-simulator expectation
- `λ ∈ (0, 1]`: multiplicative damping (depolarisation-like attenuation)
- `δ_shot`: shot-sampling noise, variance `(1 − (λE)²)/N_shots`
- `δ_gate`: additional Gaussian, variance `σ_g²`

Fit by least squares through origin on pooled (E, R) pairs: λ = (E·R)/(E·E); σ_g² from residual variance minus mean shot variance.

### Results

| Source | λ | σ_g | σ_shot | σ_total | r observed | r modelled |
|---|---:|---:|---:|---:|---:|---:|
| SV1 | **1.001** | **0.0000** | 0.030 | 0.030 | 0.994 | 0.994 |
| Singleton 4q | **0.207** | **0.1497** | 0.032 | 0.153 | 0.322 | 0.326 |
| Packed 80q (pooled) | **0.258** | **0.1593** | 0.032 | 0.162 | 0.337 | 0.376 |

### Physical interpretation

- **λ ≈ 0.21–0.26 for Rigetti**: Pauli expectation values are attenuated to ~25 % of their ideal magnitude. Consistent with a few percent per two-qubit gate error compounded across the 3-layer circuit.
- **σ_g ≈ 0.15**: ~5× the shot-noise floor. The Gaussian-residual component of error dominates shot noise.
- **SV1 fits at λ = 1.001, σ_g = 0.0000**: the "self-check" validates the fit. If SV1 fitted at, say, λ = 0.9, it would mean the model has the wrong structure. It doesn't — it's correct.

### Packed vs singleton λ difference

Packed (0.258) > singleton (0.207). Counter-intuitive if we expected packing to add crosstalk. Explanation: singleton measures Z+ZZ, whose pairwise-ZZ expectations decohere faster than single-Z. So singleton's effective λ is pulled down by the harder-to-preserve ZZ measurements. **This is not hardware evidence for packing being better** — it's a circuit-family/observable confounder.

### Diagnostic plot reading

Each histogram pair: coral = real hardware \|err\|, blue = model-synthesised \|err\|. Overlap is near-perfect for SV1 and very tight for singleton/packed. If the model were wrong, the blue histogram would have a visibly different shape — it doesn't.

### Likely questions

- **Is this a depolarising channel?** Approximately. A pure depolarising channel with strength p would give λ = (1 − 4p/3) for single-qubit Paulis — our λ = 0.26 corresponds to p ≈ 0.56 which is astronomically high. So the actual noise is more complex (asymmetric, T1/T2, readout error), but the *effect* on expectation values is well-approximated by multiplicative damping.
- **Why does σ_g exceed shot noise by 5×?** Gate errors on Rigetti superconductor are ~1–3 % per two-qubit gate. Over a 3-layer circuit with ~10 CNOTs per reservoir, cumulative error lands at ~10 %, which matches σ_g ≈ 0.15 in order of magnitude.

---

## Slide 11 — Mitigation study (Analysis 2)

### Context

Ten Ridge strategies per hardware source, comparing:
- **floor (raw_only)**: no quantum features at all
- **ceiling (exact_ideal)**: raw + noise-free quantum features
- **baseline (hw_baseline)**: raw + all hardware quantum features (current result)
- Four mitigations

### The grouped bar chart

Each group on the x-axis is one (source, size). Bars within each group are strategies. Quantum-only bars (27+) are annotated as text above each group, off the main y-axis, so the scale stays readable.

### Strategy-by-strategy

**`hw_quantum_only`**: Ridge on hardware quantum features alone, no raw. MSE 27–35 — catastrophic. Confirms that Ridge was getting essentially all its signal from the raw columns in `hw_baseline`.

**`hw_damping_corrected`**: multiply hardware features by 1/λ (undoes multiplicative bias). Helps once at packed n=100+50 (5.46 → 4.64), doesn't help at other sizes. Likely a small-sample artefact — RidgeCV α grid interacts with feature scaling; with more data, Ridge self-corrects.

**`hw_top_k`**: rank features by train-set correlation with exact (no test leakage), keep top k. k ∈ {1, 2, 4, 8}. Matches `hw_baseline` to within 0.05 MSE units. Because Ridge's α is already suppressing the noisy features — explicit selection is redundant.

### What Ridge is really doing

`hw_baseline` MSE ≈ `raw_only` MSE. Ridge's α-regularisation is setting the quantum-feature coefficients close to zero. The ~5 % improvement we see is a tiny residual signal that slips through.

### Bottom line

**No classical post-processing on the saved hardware features recovers the information lost to noise.** The gap from `hw_baseline` to `exact_ideal` is 2.6–3.1 MSE units; nothing we tried closes it by more than 0.8 (and that one improvement was fragile).

### What this means for future work

Mitigation has to happen at the **quantum-circuit level**:
- **Zero-noise extrapolation (ZNE)**: run the circuit at 1×, 2×, 3× folded depth, extrapolate to zero.
- **Probabilistic error cancellation (PEC)**: requires a calibrated device noise model; inverts known errors probabilistically.
- **Dynamical decoupling**: idle qubits do nothing between gates — add X-X pulse trains to refocus errors.
- **Shorter circuits**: fewer CNOTs → higher λ. The 3×3 reservoir could probably drop to 2 layers with marginal feature-quality loss on the simulator.

### Likely questions

- **Why did damping correction help at n=100 but not larger n?** RidgeCV α grid interaction. At n=100 Ridge has less data to pick α well, so rescaling features shifts the optimal α into a better grid point. At larger n, Ridge picks α correctly either way.
- **What about ensemble averaging of reservoirs?** Averaging makes sense only if reservoirs are estimating the same quantity — they're not, they have different random weights. Not a meaningful mitigation here.

---

## Slide 12 — Matched noise injection (Analysis 3)

### Context

Start from exact-sim quantum features. Add i.i.d. Gaussian noise of standard deviation σ to the quantum columns (raw columns untouched). Fit Ridge. Measure test MSE. Sweep σ from 0 to 0.8 in 33 steps, average 16 noise realisations per σ for stability. Result is an MSE(σ) curve per source.

σ* is the σ that matches the observed Rigetti MSE for that source — found by linear interpolation on the curve.

### Results

| Source | n_train | exact MSE | hw MSE | **σ*** |
|---|---:|---:|---:|---:|
| Singleton Z+ZZ | 100 | 2.20 | 5.51 | **0.42** |
| Packed Z | 100 | 2.84 | 5.46 | **0.46** |
| Packed Z | 500 | 2.00 | 4.07 | **0.59** |
| Packed Z | 1000 | 2.59 | 5.43 | **0.62** |

### Why σ* grows with n_train

Ridge's α is chosen by CV. With more data, Ridge has more signal to fit and can set a smaller α — more trust in the features. So even noisy features contribute more at large n, and it takes more injected noise to degrade the model to hardware's MSE.

This tells us σ* is a **task-level** noise equivalent — it depends on the downstream Ridge-on-n_samples pipeline, not just the hardware. It is not directly comparable to Analysis 1's σ_g (which was a per-measurement physical parameter).

### Relation to Analysis 1

From Analysis 1 we have λ ≈ 0.26 and σ_g ≈ 0.16 for packed Rigetti. The expected per-measurement RMS deviation from exact is:

```
σ_eff² ≈ (1 − λ)² · ⟨E²⟩ + σ_g²
       ≈ (0.74)² · 0.08 + 0.025
       ≈ 0.044 + 0.025 = 0.069
σ_eff ≈ 0.26
```

Analysis 3's σ* ≈ 0.46 (at n=100+50) is about 1.8× this. The mismatch reflects:
- σ_eff is per-measurement; σ* is per-Ridge-pipeline.
- Ridge amplifies noise through α choice and coefficient interactions.

### One-line summary

**Rigetti Ankaa-3 output, fed through our Ridge regression, degrades MSE as much as adding Gaussian noise of σ ≈ 0.5 to the exact-sim features.**

### Likely questions

- **Can we use σ* to estimate cost for a downstream experiment?** Partially. If you fix the Ridge pipeline and scale n_train, σ* tells you how the hardware's task-level impact scales. If you change the pipeline (e.g., switch to random forests), σ* changes too.
- **Why not just report σ*?** It's an easier number to communicate than (λ, σ_g), but it hides the physics. Use σ* for audience summary; use (λ, σ_g) for scientific discussion.

---

## Overall narrative for the Q&A

If someone asks "so what's the point — did it work?":

> The reservoir augmentation idea is sound — exact simulation cuts Ridge MSE by half. But on current Rigetti hardware, the depolarisation and gate-error budget is so large that almost all of that gain is destroyed. We've quantified this in two complementary ways: a physical noise model (λ=0.25, σ_g=0.15) and a task-level noise equivalent (σ*≈0.5). Classical post-processing can't recover the signal — we'd need quantum error mitigation to close the gap.

If someone asks "what would you do differently with more time/access":

> Three things, in order: (1) rerun the `pack_factor=1` singleton on the improved augmenter to cleanly test crosstalk ($540 budget); (2) apply zero-noise extrapolation at 2-3 stretch factors to measure how much λ recovers; (3) try a much shorter circuit (1-layer reservoir) — the CNOT count is the dominant source of depolarisation, so a shallower circuit on hardware might actually outperform the deeper one.
