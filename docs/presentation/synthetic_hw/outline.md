# Presentation — Speaking Outline

Your on-stage reference. Each slide section has:
- **Say** — written-form bullets (what the audience hears)
- **Oral hints** — key numbers / transitions / cues to remember at a glance

Full background is in `prep.md`; raw slide content is in `slides.md`.

---

## Slide 1 — Available AWS hardware

**Say:**
- AWS Braket lists several QPU providers (IonQ, IQM, Rigetti, D-Wave).
- For our region and tenant, most were blocked — unavailable or unreachable from the Braket console.
- Two options were consistently accessible: the SV1 state-vector simulator and Rigetti's Ankaa-3 superconducting QPU.
- Ankaa-3 has 82 qubits in a rectangular-lattice topology (show layout).
- SV1 is an on-demand simulator capable of up to 34 qubits with shot-based sampling.

**Oral hints:**
- "Ankaa-3, 82 qubits, nearest-neighbour grid."
- "SV1 — a classical simulator, still priced per task but far cheaper."
- Transition: *"Why these two and not others? — cost next."*

---

## Slide 2 — Cost constraints

**Say:**
- Braket charges per task + per shot. Ankaa-3 is $0.30 per task plus $0.0009 per shot.
- Each "task" is one circuit submission; each "shot" is one computational-basis sample.
- At 1000 shots and our reservoir circuit (3 ensembles × N samples), we scale linearly in samples.
- For 1500 samples packed (slide 3), cost is about $270 total — our biggest run.
- Anything larger than 1500 samples becomes prohibitive given our single-shot research budget.

**Oral hints:**
- "Three-hundred dollars budget for the 1500-sample run."
- "The small run at 150 samples costs $29 — that's our cheap probe."
- Transition: *"We got here by packing — next slide."*

---

## Slide 3 — Packed execution on Ankaa-3

**Say:**
- The reservoir circuit uses only 4 qubits, but Ankaa-3 has 82 available.
- We submit one Braket task that contains 20 independent reservoir circuits on disjoint 4-qubit wire sets.
- The 20 circuits share the same task-submission overhead — one $0.30 fee instead of twenty.
- So a 150-sample × 3-reservoir run that would be 450 tasks unpacked becomes only 24 packed tasks.
- Trade-off: 80 qubits are simultaneously active per task — in principle this could add crosstalk or idle-qubit error. We test this on slide 9.
- **Footnote**: earlier experiments went through PennyLane + parameter broadcasting, which silently collapsed per-observable values inside each commuting measurement group. We switched to native Braket circuits because of this. Due to time pressure, the earliest Rigetti runs use a simpler reservoir circuit than later runs.

**Oral hints:**
- "Twenty-times fewer API calls."
- "$29 packed vs ~$540 unpacked at 150 samples — same science, 18× cheaper."
- Transition: *"Here's what that bought us."*

---

## Slide 4 — Headline results

**Say:**
- This is the single most important table in this section.
- Three data sizes: 100 / 500 / 1000 train samples.
- For each size, three methods: Raw features only (floor), noiseless simulation (ceiling), Rigetti hardware with 1000 shots.
- The noiseless simulator is exactly what we'd want — it cuts Ridge MSE by about half at every size.
- Real Rigetti hardware captures only a few percent of that gain: 1 % at n=500, 6 % at n=100, never more.
- The residual standard deviation (last column) for hardware matches raw closely — same magnitude of prediction error.

**Oral hints:**
- "Half the MSE with exact sim; at most six percent with Rigetti."
- "Raw-only is our floor, noiseless sim our ceiling."
- Transition: *"Something is clearly wrong at the hardware end. Let's zoom in."*

---

## Slide 5 — MSE visualisation

**Say:**
- Placeholder — the old line plots were uninformative. A grouped bar chart (method × size) will go here.

**Oral hints:**
- Skip or briefly acknowledge; defer to slide 4's table for now.

---

## Slide 6 — Per-feature fidelity

**Say:**
- We have 12 augmented features — 3 reservoirs × 4 qubits each.
- For each feature, compute the Pearson correlation between exact-sim output and hardware output, at every data size.
- Perfect hardware would give correlation 1 on every line.
- We see a spread between 0.05 and 0.65. Most features drift between 0.1 and 0.5 — partial information survives.
- Two features stand out above 0.4 consistently: reservoir-1 qubit-0 and reservoir-2 qubit-0.
- The ranking is fairly stable across data sizes — it's a property of the circuit on this hardware, not a statistical fluke.

**Oral hints:**
- "Average correlation around 0.3 — not random, not good."
- "Some features survive better — at least one per reservoir around 0.4–0.5."
- Transition: *"What does that mean at the level of individual measurements?"*

---

## Slide 7 — Per-measurement scatter

**Say:**
- Each point is one (sample, feature) measurement: exact sim on x, hardware on y.
- If hardware were noise-free, every point would sit on the dashed identity line.
- What we see is a broad, roughly isotropic cloud centred on the line with mean absolute error ~0.2.
- Pooled correlation: 0.33–0.39 — matches the per-feature plot averaged over features.
- Crucially, the cloud shape does **not** shrink as we add more data. More samples just fill in the cloud more densely.
- This tells us: noise is per-circuit, per-shot — not something that averages out by running more samples.

**Oral hints:**
- "Mean error of 0.2 — that's 20 % of full Pauli range."
- "Cloud is the same shape at 150 and 1500 samples — shot averaging doesn't save us."
- Transition: *"How much of this is shot noise vs hardware noise?"*

---

## Slide 8 — Error histogram and shot-noise floor

**Say:**
- Error = |exact − source| for every measurement, plotted as a probability density.
- Green histogram is SV1 — a state-vector simulator with only shot sampling, no hardware noise.
- SV1 sits on the analytic shot-noise floor at 0.030 (dashed line).
- The three Rigetti histograms (singleton 4q, packed 80q at two sizes) overlap each other and peak around 0.05–0.10, with long tails out past 0.8.
- Mean error for Rigetti is around 0.20 vs 0.024 for SV1 — a factor of seven.
- So only about 15 % of the hardware error budget is shot noise. The remaining 85 % is gate, readout, and crosstalk error.

**Oral hints:**
- "SV1 sits on shot noise, Rigetti sits seven times higher."
- "More shots won't fix this — we'd need better gates."
- Transition: *"Does packing density matter in any of this?"*

---

## Slide 9 — Three-way overlay — packing crosstalk test

**Say:**
- Two kinds of hardware runs: singleton at 4q active density, packed at 80q active density.
- Both plotted as (exact, hardware) scatter on shared axes.
- Singleton mean error is 0.200; packed is 0.210. Difference is within noise of the measurement.
- **Naive expectation was**: packing 80 qubits together would add crosstalk and make things worse. That's not what we see.
- **Caveat — shown on the figure**: the two runs use different reservoir circuits because time pressure forced the singleton run through an earlier augmenter implementation. Observable mix also differs (Z+ZZ vs Z-only). So this is not a perfectly controlled test.
- The important qualitative read: packing density does not add a dominant extra error term on top of per-circuit depolarisation.

**Oral hints:**
- "Packed is not meaningfully worse than singleton."
- "The two curves have comparable spread — no obvious crosstalk."
- "Caveat: different circuits, so treat this as qualitative."
- Transition: *"Can we explain all of this with a simple noise model?"*

---

## Slide 10 — Noise-model calibration (Analysis 1)

**Say:**
- Fit a two-parameter model to each source: hardware output = λ × exact + shot noise + Gaussian gate noise σ_g.
- λ ∈ (0,1] is multiplicative damping — like depolarisation attenuating the expectation value toward zero.
- σ_g is additional Gaussian noise beyond shot noise.
- SV1 fits at λ = 1.001 and σ_g = 0 — exactly what you'd expect for shot-only sampling. This validates the fit methodology.
- Singleton (4q): λ ≈ 0.21, σ_g ≈ 0.150 — strong damping and 5× the shot-noise floor in Gaussian error.
- Packed (80q, pooled across sizes): λ ≈ 0.26, σ_g ≈ 0.159 — nearly identical to singleton.
- Model predicts hardware correlation within 1–4 % of observed — good fit.
- **Plot**: blue bars are model-synthesised error, coral bars are real hardware — they overlap almost perfectly.

**Oral hints:**
- "Rigetti returns about 25 % of the ideal expectation value — that's severe depolarisation."
- "Gate noise of 0.15 is five times shot noise — the limiting factor."
- "SV1 validates the fit — λ = 1, σ_g = 0 exactly."
- Transition: *"Given this, can we recover the lost signal classically?"*

---

## Slide 11 — Mitigation study (Analysis 2)

**Say:**
- Baseline: Ridge on raw + hardware quantum features — MSE 4.07–5.46 depending on size.
- Ceiling: raw + exact-sim quantum — MSE 2.00–2.84.
- Quantum-only: Ridge on hardware quantum features alone, no raw — MSE blows up past 27. Hardware features alone cannot predict Y at all.
- Damping correction: multiply hardware features by 1/λ to undo the attenuation. Helps once, at packed n=100+50, then does nothing.
- Top-k feature selection: rank features by train-set correlation with exact sim, keep top 1/2/4/8. Matches the baseline exactly — Ridge's built-in regularisation is already doing this implicitly.
- **Bottom line**: no classical post-processing on existing hardware features meaningfully closes the gap to the ceiling. The information is gone.

**Oral hints:**
- "Nothing classical helps."
- "Quantum-only blows up — Ridge has nothing to work with without raw."
- "This means error mitigation has to happen at the quantum level: ZNE, PEC, dynamical decoupling."
- Transition: *"One last summary, as a single number."*

---

## Slide 12 — Matched noise injection (Analysis 4)

**Say:**
- Take the exact-sim quantum features. Add i.i.d. Gaussian noise of standard deviation σ. Sweep σ from 0 to 0.8. Re-fit Ridge at each σ.
- For each σ we get a classical noisy-features MSE. The σ* that matches the observed Rigetti MSE is the "noise-equivalent sigma".
- σ* comes out between 0.42 and 0.62 across configurations.
- σ* grows with training size — at larger n_train, Ridge suppresses noisy features more effectively, so we need more injected noise to reach the same MSE.
- One-number summary: Rigetti Ankaa-3 degrades Ridge-task MSE the same way as adding Gaussian noise of σ ≈ 0.5 to the exact features.
- This is a task-level complement to Analysis 1's physical (λ, σ_g) decomposition.

**Oral hints:**
- "Half-unit Gaussian noise on the exact features — that's Rigetti for Ridge."
- "σ* grows with n_train — not a pure hardware number."
- Closing: *"Everything we do downstream has to work with this σ ≈ 0.5 equivalent. Without quantum-level error mitigation, we can't do better."*
