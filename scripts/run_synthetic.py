"""Run the full synthetic regime-switching experiment."""

from src.synthetic.config import (
    AugmenterConfig,
    DGPConfig,
    ExperimentConfig,
    ModelConfig,
)
from src.synthetic.runner import ExperimentRunner


# ── Classical baselines (Group A) ──────────────────────────────────────────
CLASSICAL_AUGMENTERS = [
    AugmenterConfig("identity", "classical"),
    AugmenterConfig("poly_deg2", "classical"),
    AugmenterConfig("poly_deg3", "classical"),
    AugmenterConfig("interaction_log", "classical"),
    AugmenterConfig("rff_10", "classical", {"n_components": 10, "gamma": 0.5}),
    AugmenterConfig("rff_30", "classical", {"n_components": 30, "gamma": 0.5}),
]

# ── Quantum one-shot (Group B) ────────────────────────────────────────────
QUANTUM_FIXED_AUGMENTERS = [
    AugmenterConfig("angle_basic_2L", "quantum_fixed", {"n_layers": 2}),
    AugmenterConfig("angle_strong_2L", "quantum_fixed", {"n_layers": 2}),
    AugmenterConfig("angle_strong_3L", "quantum_fixed", {"n_layers": 3}),
    AugmenterConfig("zz_reps2", "quantum_fixed", {"reps": 2}),
    AugmenterConfig("iqp_reps3", "quantum_fixed", {"n_repeats": 3}),
    AugmenterConfig("qaoa_p2", "quantum_fixed", {"p": 2}),
    AugmenterConfig("prob_2L", "quantum_fixed", {"n_layers": 2}),
]

# ── Quantum reservoirs (Group C) ──────────────────────────────────────────
RESERVOIR_AUGMENTERS = [
    AugmenterConfig("reservoir_3x3", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3}),
    AugmenterConfig("reservoir_3x3_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "measure_pairwise": True}),
]

# ── Learning-based (Group D) ──────────────────────────────────────────────
LEARNED_AUGMENTERS = [
    AugmenterConfig("mlp_10", "neural", {"n_out": 10, "hidden_dims": (32, 16)}),
    AugmenterConfig("mlp_shallow_10", "neural", {"n_out": 10, "hidden_dims": (10,)}),
    AugmenterConfig("autoencoder_4", "neural", {"bottleneck_dim": 4}),
    AugmenterConfig("learned_rff_10", "neural", {"n_components": 10}),
    AugmenterConfig("vqc_strong_2L", "quantum_learned", {"n_layers": 2, "n_epochs": 200, "train_subsample": 500}),
]

# ── Diagnostics (Group E) ─────────────────────────────────────────────────
DIAGNOSTIC_AUGMENTERS = [
    AugmenterConfig("oracle", "classical"),
]

# ── Models ────────────────────────────────────────────────────────────────
MODELS = [
    ModelConfig("ols"),
    ModelConfig("ridge"),
    ModelConfig("lasso"),
]


def run_classical_only():
    """Quick run: classical baselines + oracle only."""
    config = ExperimentConfig(
        dgp=DGPConfig(),
        augmenters=CLASSICAL_AUGMENTERS + DIAGNOSTIC_AUGMENTERS,
        models=MODELS,
        seeds=[42, 123, 456],
    )
    runner = ExperimentRunner(config)
    return runner.run()


def run_full():
    """Full experiment: all augmenters × all models × 5 seeds."""
    config = ExperimentConfig(
        dgp=DGPConfig(),
        augmenters=(
            CLASSICAL_AUGMENTERS
            + QUANTUM_FIXED_AUGMENTERS
            + RESERVOIR_AUGMENTERS
            + LEARNED_AUGMENTERS
            + DIAGNOSTIC_AUGMENTERS
        ),
        models=MODELS,
        seeds=[42, 123, 456, 789, 1024],
    )
    runner = ExperimentRunner(config)
    return runner.run()


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "classical"

    if mode == "classical":
        print("Running classical baselines only...")
        run_classical_only()
    elif mode == "full":
        print("Running full experiment (this will take a while)...")
        run_full()
    else:
        print(f"Usage: python -m scripts.run_synthetic [classical|full]")
        sys.exit(1)
