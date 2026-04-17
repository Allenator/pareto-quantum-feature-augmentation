"""Run the real financial data walk-forward backtest experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthetic.config import AugmenterConfig, ModelConfig
from src.real.config import RealDataConfig, BacktestConfig, ExperimentConfig
from src.real.backtest import BacktestRunner

# ── Classical baselines ───────────────────────────────────────────────────
CLASSICAL = [
    AugmenterConfig("identity", "classical"),
    AugmenterConfig("poly_deg2_interact", "classical"),
    AugmenterConfig("poly_deg2", "classical"),
    AugmenterConfig("interaction_log", "classical"),
    AugmenterConfig("rff_10", "classical", {"n_components": 10, "gamma": 0.5}),
    AugmenterConfig("rff_30", "classical", {"n_components": 30, "gamma": 0.5}),
    AugmenterConfig("rff_50", "classical", {"n_components": 50, "gamma": 0.5}),
    AugmenterConfig("rff_96", "classical", {"n_components": 96, "gamma": 0.5}),
]

# ── Quantum reservoir augmenters (simple: RY + linear CNOT + Z-only) ─────
# QuantumReservoir from scripts/quantum_reservoir.py (RY → [Rot+CNOT]×L → Rot → ⟨Zᵢ⟩)
# Uses "qres" prefix to route through QuantumReservoirAugmenter wrapper.
# n_qubits must equal input feature dimension (14) for this augmenter.
QUANTUM_RESERVOIR = [
    # Ensemble size sweep at 14 qubits (1:1 feature mapping)
    AugmenterConfig("qres_3x2_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 2, "n_ensemble": 3}),
    AugmenterConfig("qres_5x2_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 2, "n_ensemble": 5}),
    AugmenterConfig("qres_3x3_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qres_5x3_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 3, "n_ensemble": 5}),
    # Depth sweep
    AugmenterConfig("qres_3x1_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 1, "n_ensemble": 3}),
    AugmenterConfig("qres_3x5_14q", "quantum_fixed",
        {"n_qubits": 14, "n_layers": 5, "n_ensemble": 3}),
]

N_FEATURES = 14  # len(FEATURE_COLS)

# Common kwargs for the winning synthetic design axes
_UNIFIED_BASE = dict(
    n_features=N_FEATURES, encoding="angle", connectivity="circular",
    cnot_mixing=True, random_rot=True,
)

# ── Unified quantum augmenters (angle + circular CNOT + configurable obs) ─
# Uses "qunified" prefix to route through UnifiedReservoirAugmenter.
# Modular feature-to-qubit mapping (averages features into qubit bins).
QUANTUM_UNIFIED = [
    # Main sweep: qubit count × observables
    AugmenterConfig("qunified_z_6q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 6, "observables": "Z", "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qunified_z_8q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z", "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qunified_xyz_6q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 6, "observables": "XYZ", "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qunified_xyz_8q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "XYZ", "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qunified_zzz_6q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 6, "observables": "Z+ZZ", "n_layers": 3, "n_ensemble": 3}),
    AugmenterConfig("qunified_zzz_8q_3L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z+ZZ", "n_layers": 3, "n_ensemble": 3}),
    # PCA mapping variant
    AugmenterConfig("qunified_z_8q_3L_3ens_pca", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z", "n_layers": 3, "n_ensemble": 3,
         "qubit_mapping": "pca"}),
    # Ablations: depth
    AugmenterConfig("qunified_z_8q_2L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z", "n_layers": 2, "n_ensemble": 3}),
    AugmenterConfig("qunified_z_8q_5L_3ens", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z", "n_layers": 5, "n_ensemble": 3}),
    # Ablation: connectivity
    AugmenterConfig("qunified_z_8q_3L_3ens_linear", "quantum_fixed",
        {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z", "n_layers": 3, "n_ensemble": 3,
         "connectivity": "linear"}),
]

QUANTUM = QUANTUM_UNIFIED  # qres excluded — slow per-sample loop at 14 qubits

MODELS = [ModelConfig("ridge")]

SEEDS_PLACEHOLDER = [42]  # Real data doesn't need seed variation (data is fixed)


def run_quick():
    """Quick test: 3 tickers, monthly eval steps, minimal augmenters."""
    config = ExperimentConfig(
        data=RealDataConfig(
            tickers=("AAPL", "MSFT", "NVDA"),
            start_date="2022-01-01",
            end_date="2025-12-31",
        ),
        backtest=BacktestConfig(step_days=21),  # monthly
        augmenters=[
            AugmenterConfig("identity", "classical"),
            AugmenterConfig("poly_deg2", "classical"),
            AugmenterConfig("qunified_z_8q_3L_3ens", "quantum_fixed",
                {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z",
                 "n_layers": 3, "n_ensemble": 3}),
        ],
        models=[ModelConfig("ridge")],
        run_id="quick",
    )
    runner = BacktestRunner(config)
    return runner.run()


def run_monthly():
    """All augmenters, 10 tickers, monthly eval steps."""
    config = ExperimentConfig(
        data=RealDataConfig(),
        backtest=BacktestConfig(step_days=21),  # monthly
        augmenters=CLASSICAL + QUANTUM,
        models=MODELS,
        run_id="monthly",
    )
    runner = BacktestRunner(config)
    return runner.run()


def run_full():
    """Full experiment: 10 tickers, daily eval, all augmenters."""
    config = ExperimentConfig(
        data=RealDataConfig(),
        backtest=BacktestConfig(),
        augmenters=CLASSICAL + QUANTUM,
        models=MODELS,
        run_id="full",
    )
    runner = BacktestRunner(config)
    return runner.run()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    if mode == "quick":
        run_quick()
    elif mode == "monthly":
        run_monthly()
    elif mode == "full":
        run_full()
    else:
        print(f"Usage: python scripts/run_real.py [quick|monthly|full]")
        sys.exit(1)
