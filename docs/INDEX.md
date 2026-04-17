# Documentation Index

All documentation in `docs/` is listed here. Update this index whenever documentation is added, removed, or significantly changed.

## Specifications

- [Challenge Specification](specs/challenge.md) — Full YQuantum 2026 AWS x State Street challenge requirements, tasks, metrics, and evaluation criteria

## Designs

- [Synthetic Experiment Design](designs/synthetic.md) — Architecture, augmenter inventory, fairness controls, and initial results for the synthetic regime-switching experiment
- [Unified Quantum Design](designs/unified_quantum_design.md) — Factorial design strategy for systematically comparing quantum feature maps across 7 independent dimensions
- [Preprocessing Pipeline](designs/preprocessing.md) — Data generation, standardization, clipping, and fairness controls
- [Classical Baselines](designs/classical_baselines.md) — Classical feature augmentation methods: polynomial, log/abs, RFF, oracle, and comparison table
- [Quantum Reservoir](designs/quantum_reservoir.md) — Optimal quantum feature augmentation strategy: circuit structure, design justification, usage, and integration
- [Real Financial Data Design](designs/real.md) — DataBento pipeline, walk-forward backtesting with cached augmentation, unified quantum augmenter, cross-asset extensions, and S&P 500 excess return prediction
- [Synthetic-Data Hardware Benchmark](designs/synthetic_hw.md) — Figures comparing `reservoir_3x3` on Rigetti Ankaa-3 (packed vs unpacked) against exact simulator, with Ridge-MSE training results and per-feature fidelity analysis
- [Synthetic-Data Hardware — Data Inventory](designs/synthetic_hw_data.md) — Authoritative index of every saved feature file in `features/synthetic_hw/`, the filename convention, variant glossary, and verified (exact, hardware) input-alignment pairs

## Results

- [Exploration Results](../exploration/results.md) — Initial run results for all exploration scripts: baselines, quantum feature maps, hybrid workflows, evaluation comparison, real stock data pipeline
- [Synthetic-Data Hardware Results](results/synthetic_hw.md) — Noise-model calibration, mitigation study, and matched noise-injection analysis comparing Rigetti Ankaa-3 / SV1 / exact simulator on the `reservoir_3x3` augmenter

## Presentation

### Synthetic-data hardware-experiment section

Scope: procedure, analysis, and conclusions for the part that experiments on real quantum machines (Rigetti Ankaa-3 + SV1). Other presentation sections (quantum/classical setup, etc.) are authored separately by other team members.

- [Slides](presentation/synthetic_hw/slides.md) — 12-slide deck: hardware availability, cost, packing, headline results, feature fidelity, noise model, mitigation, matched noise injection. Spare by design (figure + title + one-liner).
- [Speaking outline](presentation/synthetic_hw/outline.md) — On-stage quick reference; per-slide bullets + oral hints with key numbers and transitions.
- [Preparation document](presentation/synthetic_hw/prep.md) — Comprehensive read-before-presenting doc with context, numbers, and likely Q&A per slide.

## Changelog

- [Session changelog — synthetic hardware benchmark](CHANGES.md) — New scripts, file renames, deletions, and documentation added while analysing existing Rigetti / SV1 / exact-sim feature files (no new hardware runs).
