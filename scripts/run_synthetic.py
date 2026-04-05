"""Run the full synthetic regime-switching experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.config import (
    AugmenterConfig,
    DGPConfig,
    ExperimentConfig,
    ModelConfig,
)
from src.synthetic.runner import ExperimentRunner

# ── Classical baselines ───────────────────────────────────────────────────
CLASSICAL = [
    AugmenterConfig("identity", "classical"),
    AugmenterConfig("oracle", "classical"),
    AugmenterConfig("poly_deg2_interact", "classical"),
    AugmenterConfig("poly_deg2", "classical"),
    AugmenterConfig("poly_deg3", "classical"),
    AugmenterConfig("interaction_log", "classical"),
    AugmenterConfig("rff_6", "classical", {"n_components": 6, "gamma": 0.5}),
    AugmenterConfig("rff_10", "classical", {"n_components": 10, "gamma": 0.5}),
    AugmenterConfig("rff_15", "classical", {"n_components": 15, "gamma": 0.5}),
    AugmenterConfig("rff_21", "classical", {"n_components": 21, "gamma": 0.5}),
    AugmenterConfig("rff_30", "classical", {"n_components": 30, "gamma": 0.5}),
    AugmenterConfig("rff_36", "classical", {"n_components": 36, "gamma": 0.5}),
    AugmenterConfig("rff_50", "classical", {"n_components": 50, "gamma": 0.5}),
    AugmenterConfig("rff_96", "classical", {"n_components": 96, "gamma": 0.5}),
]

# ── Angle encoding sweep (StronglyEntangling, vary n_qubits) ─────────────
ANGLE_SWEEP = [
    AugmenterConfig("angle_strong_3q_2L", "quantum_fixed", {"n_qubits": 3, "n_layers": 2}),
    AugmenterConfig("angle_strong_4q_2L", "quantum_fixed", {"n_qubits": 4, "n_layers": 2}),
    AugmenterConfig("angle_strong_5q_2L", "quantum_fixed", {"n_qubits": 5, "n_layers": 2}),
    AugmenterConfig("angle_strong_6q_2L", "quantum_fixed", {"n_qubits": 6, "n_layers": 2}),
    AugmenterConfig("angle_strong_7q_2L", "quantum_fixed", {"n_qubits": 7, "n_layers": 2}),
    AugmenterConfig("angle_strong_8q_2L", "quantum_fixed", {"n_qubits": 8, "n_layers": 2}),
    AugmenterConfig("angle_strong_9q_2L", "quantum_fixed", {"n_qubits": 9, "n_layers": 2}),
    AugmenterConfig("angle_strong_10q_2L", "quantum_fixed", {"n_qubits": 10, "n_layers": 2}),
    AugmenterConfig("angle_strong_13q_2L", "quantum_fixed", {"n_qubits": 13, "n_layers": 2}),
    # Depth sweep at fixed n_qubits=4
    AugmenterConfig("angle_strong_4q_1L", "quantum_fixed", {"n_qubits": 4, "n_layers": 1}),
    AugmenterConfig("angle_strong_4q_3L", "quantum_fixed", {"n_qubits": 4, "n_layers": 3}),
]

# ── ZZ feature map sweep ─────────────────────────────────────────────────
ZZ_SWEEP = [
    AugmenterConfig("zz_3q_r2", "quantum_fixed", {"n_qubits": 3, "reps": 2}),
    AugmenterConfig("zz_4q_r2", "quantum_fixed", {"n_qubits": 4, "reps": 2}),
    AugmenterConfig("zz_5q_r2", "quantum_fixed", {"n_qubits": 5, "reps": 2}),
    AugmenterConfig("zz_6q_r2", "quantum_fixed", {"n_qubits": 6, "reps": 2}),
    AugmenterConfig("zz_7q_r2", "quantum_fixed", {"n_qubits": 7, "reps": 2}),
    AugmenterConfig("zz_8q_r2", "quantum_fixed", {"n_qubits": 8, "reps": 2}),
    AugmenterConfig("zz_9q_r2", "quantum_fixed", {"n_qubits": 9, "reps": 2}),
    AugmenterConfig("zz_10q_r2", "quantum_fixed", {"n_qubits": 10, "reps": 2}),
    AugmenterConfig("zz_13q_r2", "quantum_fixed", {"n_qubits": 13, "reps": 2}),
]

# ── IQP sweep ─────────────────────────────────────────────────────────────
IQP_SWEEP = [
    AugmenterConfig("iqp_3q_r3", "quantum_fixed", {"n_qubits": 3, "n_repeats": 3}),
    AugmenterConfig("iqp_4q_r3", "quantum_fixed", {"n_qubits": 4, "n_repeats": 3}),
    AugmenterConfig("iqp_5q_r3", "quantum_fixed", {"n_qubits": 5, "n_repeats": 3}),
    AugmenterConfig("iqp_6q_r3", "quantum_fixed", {"n_qubits": 6, "n_repeats": 3}),
    AugmenterConfig("iqp_7q_r3", "quantum_fixed", {"n_qubits": 7, "n_repeats": 3}),
    AugmenterConfig("iqp_8q_r3", "quantum_fixed", {"n_qubits": 8, "n_repeats": 3}),
    AugmenterConfig("iqp_9q_r3", "quantum_fixed", {"n_qubits": 9, "n_repeats": 3}),
    AugmenterConfig("iqp_10q_r3", "quantum_fixed", {"n_qubits": 10, "n_repeats": 3}),
    AugmenterConfig("iqp_13q_r3", "quantum_fixed", {"n_qubits": 13, "n_repeats": 3}),
]

# ── QAOA sweep ────────────────────────────────────────────────────────────
QAOA_SWEEP = [
    AugmenterConfig("qaoa_3q_p2", "quantum_fixed", {"n_qubits": 3, "p": 2}),
    AugmenterConfig("qaoa_4q_p2", "quantum_fixed", {"n_qubits": 4, "p": 2}),
    AugmenterConfig("qaoa_5q_p2", "quantum_fixed", {"n_qubits": 5, "p": 2}),
    AugmenterConfig("qaoa_6q_p2", "quantum_fixed", {"n_qubits": 6, "p": 2}),
    AugmenterConfig("qaoa_7q_p2", "quantum_fixed", {"n_qubits": 7, "p": 2}),
    AugmenterConfig("qaoa_8q_p2", "quantum_fixed", {"n_qubits": 8, "p": 2}),
    AugmenterConfig("qaoa_9q_p2", "quantum_fixed", {"n_qubits": 9, "p": 2}),
    AugmenterConfig("qaoa_10q_p2", "quantum_fixed", {"n_qubits": 10, "p": 2}),
    AugmenterConfig("qaoa_13q_p2", "quantum_fixed", {"n_qubits": 13, "p": 2}),
]

# ── Reservoir sweep (Z-only, vary count) ──────────────────────────────────
RESERVOIR_Z = [
    AugmenterConfig("reservoir_1x3", "quantum_fixed", {"n_reservoirs": 1, "n_layers": 3}),
    AugmenterConfig("reservoir_2x3", "quantum_fixed", {"n_reservoirs": 2, "n_layers": 3}),
    AugmenterConfig("reservoir_3x3", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3}),
    AugmenterConfig("reservoir_5x3", "quantum_fixed", {"n_reservoirs": 5, "n_layers": 3}),
    AugmenterConfig("reservoir_7x3", "quantum_fixed", {"n_reservoirs": 7, "n_layers": 3}),
    AugmenterConfig("reservoir_10x3", "quantum_fixed", {"n_reservoirs": 10, "n_layers": 3}),
    AugmenterConfig("reservoir_13x3", "quantum_fixed", {"n_reservoirs": 13, "n_layers": 3}),
    AugmenterConfig("reservoir_24x3", "quantum_fixed", {"n_reservoirs": 24, "n_layers": 3}),
]

# ── Reservoir sweep (Z+ZZ, vary count) ────────────────────────────────────
RESERVOIR_ZZ = [
    AugmenterConfig("reservoir_1x3_zz", "quantum_fixed", {"n_reservoirs": 1, "n_layers": 3, "observables": "Z+ZZ"}),
    AugmenterConfig("reservoir_2x3_zz", "quantum_fixed", {"n_reservoirs": 2, "n_layers": 3, "observables": "Z+ZZ"}),
    AugmenterConfig("reservoir_3x3_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ"}),
    AugmenterConfig("reservoir_5x3_zz", "quantum_fixed", {"n_reservoirs": 5, "n_layers": 3, "observables": "Z+ZZ"}),
    AugmenterConfig("reservoir_9x3_zz", "quantum_fixed", {"n_reservoirs": 9, "n_layers": 3, "observables": "Z+ZZ"}),
]

# ── Reservoir basis variants (3 reservoirs) ───────────────────────────────
RESERVOIR_BASIS = [
    AugmenterConfig("reservoir_3x3_X", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "X"}),
    AugmenterConfig("reservoir_3x3_Y", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Y"}),
    AugmenterConfig("reservoir_3x3_XYZ", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "XYZ"}),
    AugmenterConfig("reservoir_3x3_XYZ_ZZ", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "XYZ+ZZ"}),
    AugmenterConfig("reservoir_3x3_full", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "full"}),
    # Scale up best basis variants
    AugmenterConfig("reservoir_1x3_XYZ_ZZ", "quantum_fixed", {"n_reservoirs": 1, "n_layers": 3, "observables": "XYZ+ZZ"}),
    AugmenterConfig("reservoir_2x3_XYZ_ZZ", "quantum_fixed", {"n_reservoirs": 2, "n_layers": 3, "observables": "XYZ+ZZ"}),
    AugmenterConfig("reservoir_5x3_XYZ_ZZ", "quantum_fixed", {"n_reservoirs": 5, "n_layers": 3, "observables": "XYZ+ZZ"}),
    # 5x3 XYZ+ZZ = 94 features (already at limit)
]

# ── Reservoir entanglement + data reuploading variants ────────────────────
RESERVOIR_ADVANCED = [
    AugmenterConfig("reservoir_3x3_circ_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ", "entanglement": "circular"}),
    AugmenterConfig("reservoir_3x3_all_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ", "entanglement": "all"}),
    AugmenterConfig("reservoir_3x3_reup_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ", "data_reuploading": True}),
    AugmenterConfig("reservoir_3x3_circ_reup_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ", "entanglement": "circular", "data_reuploading": True}),
]

# ── Probability sweep ────────────────────────────────────────────────────
PROB_SWEEP = [
    AugmenterConfig("prob_3q_2L", "quantum_fixed", {"n_qubits": 3, "n_layers": 2}),
    AugmenterConfig("prob_4q_2L", "quantum_fixed", {"n_qubits": 4, "n_layers": 2}),
    AugmenterConfig("prob_5q_2L", "quantum_fixed", {"n_qubits": 5, "n_layers": 2}),
    AugmenterConfig("prob_6q_2L", "quantum_fixed", {"n_qubits": 6, "n_layers": 2}),
]

# ── All static augmenters ────────────────────────────────────────────────
ALL_STATIC = (
    CLASSICAL
    + ANGLE_SWEEP
    + ZZ_SWEEP
    + IQP_SWEEP
    + QAOA_SWEEP
    + RESERVOIR_Z
    + RESERVOIR_ZZ
    + RESERVOIR_BASIS
    + RESERVOIR_ADVANCED
    + PROB_SWEEP
)

SEEDS = [42, 123, 456, 789, 1024]


def run_static_sweep():
    """Full static augmenter sweep: all configs x Ridge+Lasso x 5 seeds."""
    config = ExperimentConfig(
        dgp=DGPConfig(),
        augmenters=ALL_STATIC,
        models=[ModelConfig("ridge"), ModelConfig("lasso")],
        seeds=SEEDS,
        run_id="static_sweep",
    )
    runner = ExperimentRunner(config)
    return runner.run()


def run_classical_only():
    """Quick run: classical baselines only."""
    config = ExperimentConfig(
        dgp=DGPConfig(),
        augmenters=CLASSICAL,
        models=[ModelConfig("ridge")],
        seeds=SEEDS,
        run_id="static_sweep",
    )
    runner = ExperimentRunner(config)
    return runner.run()


# ── Trainable augmenters ──────────────────────────────────────────────────

TRAINABLE_MODELS = [
    ModelConfig("ols"),
    ModelConfig("ridge", alpha_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
    ModelConfig("lasso"),
]

# VQC sweep (n_qubits*(n_qubits+1)/2 features via Z+ZZ)
VQC_SWEEP = [
    AugmenterConfig("vqc_4q_2L", "quantum_learned", {"n_qubits": 4, "n_layers": 2, "n_epochs": 200, "train_subsample": 1000, "batch_size": 64}),
    AugmenterConfig("vqc_5q_2L", "quantum_learned", {"n_qubits": 5, "n_layers": 2, "n_epochs": 200, "train_subsample": 1000, "batch_size": 64}),
    AugmenterConfig("vqc_6q_2L", "quantum_learned", {"n_qubits": 6, "n_layers": 2, "n_epochs": 200, "train_subsample": 1000, "batch_size": 64}),
    AugmenterConfig("vqc_7q_2L", "quantum_learned", {"n_qubits": 7, "n_layers": 2, "n_epochs": 200, "train_subsample": 1000, "batch_size": 64}),
    AugmenterConfig("vqc_8q_2L", "quantum_learned", {"n_qubits": 8, "n_layers": 2, "n_epochs": 100, "train_subsample": 500, "batch_size": 64}),
]

# MLP sweep
MLP_SWEEP = [
    AugmenterConfig("mlp_6", "neural", {"n_out": 6, "hidden_dims": (16,), "n_epochs": 1000}),
    AugmenterConfig("mlp_10", "neural", {"n_out": 10, "hidden_dims": (32, 16), "n_epochs": 1000}),
    AugmenterConfig("mlp_15", "neural", {"n_out": 15, "hidden_dims": (32, 16), "n_epochs": 1000}),
    AugmenterConfig("mlp_21", "neural", {"n_out": 21, "hidden_dims": (64, 32), "n_epochs": 1000}),
    AugmenterConfig("mlp_30", "neural", {"n_out": 30, "hidden_dims": (64, 32), "n_epochs": 1000}),
    AugmenterConfig("mlp_50", "neural", {"n_out": 50, "hidden_dims": (128, 64), "n_epochs": 1000}),
]

# Learned RFF sweep
LEARNED_RFF_SWEEP = [
    AugmenterConfig("learned_rff_6", "neural", {"n_components": 6, "n_epochs": 1000}),
    AugmenterConfig("learned_rff_10", "neural", {"n_components": 10, "n_epochs": 1000}),
    AugmenterConfig("learned_rff_15", "neural", {"n_components": 15, "n_epochs": 1000}),
    AugmenterConfig("learned_rff_21", "neural", {"n_components": 21, "n_epochs": 1000}),
    AugmenterConfig("learned_rff_30", "neural", {"n_components": 30, "n_epochs": 1000}),
    AugmenterConfig("learned_rff_50", "neural", {"n_components": 50, "n_epochs": 1000}),
]

# Autoencoder sweep
AE_SWEEP = [
    AugmenterConfig("autoencoder_4", "neural", {"bottleneck_dim": 4, "hidden_dim": 16, "n_epochs": 1000}),
    AugmenterConfig("autoencoder_8", "neural", {"bottleneck_dim": 8, "hidden_dim": 32, "n_epochs": 1000}),
    AugmenterConfig("autoencoder_16", "neural", {"bottleneck_dim": 16, "hidden_dim": 64, "n_epochs": 1000}),
    AugmenterConfig("autoencoder_32", "neural", {"bottleneck_dim": 32, "hidden_dim": 128, "n_epochs": 1000}),
]

ALL_TRAINABLE = VQC_SWEEP + MLP_SWEEP + LEARNED_RFF_SWEEP + AE_SWEEP


def run_trainable_sweep():
    """Trainable augmenter sweep: all configs x OLS/Ridge/Lasso x 5 seeds."""
    # Include static references for comparison in the same results dir
    references = [
        AugmenterConfig("identity", "classical"),
        AugmenterConfig("oracle", "classical"),
        AugmenterConfig("rff_10", "classical", {"n_components": 10, "gamma": 0.5}),
        AugmenterConfig("rff_50", "classical", {"n_components": 50, "gamma": 0.5}),
        AugmenterConfig("reservoir_3x3_zz", "quantum_fixed", {"n_reservoirs": 3, "n_layers": 3, "observables": "Z+ZZ"}),
    ]
    config = ExperimentConfig(
        dgp=DGPConfig(),
        augmenters=references + ALL_TRAINABLE,
        models=TRAINABLE_MODELS,
        seeds=SEEDS,
        run_id="trainable_sweep",
    )
    runner = ExperimentRunner(config)
    return runner.run()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "static"

    if mode == "classical":
        run_classical_only()
    elif mode == "static":
        run_static_sweep()
    elif mode == "trainable":
        run_trainable_sweep()
    else:
        print(f"Usage: python scripts/run_synthetic.py [classical|static|trainable]")
        sys.exit(1)
