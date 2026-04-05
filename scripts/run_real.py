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
]

# ── Quantum augmenters ────────────────────────────────────────────────────
# QuantumReservoir from scripts/quantum_reservoir.py (RY → [Rot+CNOT]×L → Rot → ⟨Zᵢ⟩)
# Uses "qres" prefix to route through QuantumReservoirAugmenter wrapper.
# n_qubits must equal input feature dimension (14) for this augmenter.
QUANTUM = [
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

MODELS = [ModelConfig("ridge"), ModelConfig("lasso")]

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
            AugmenterConfig("qres_3x2_14q", "quantum_fixed",
                {"n_qubits": 14, "n_layers": 2, "n_ensemble": 3}),
        ],
        models=[ModelConfig("ridge")],
        run_id="quick",
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
    elif mode == "full":
        run_full()
    else:
        print(f"Usage: python scripts/run_real.py [quick|full]")
        sys.exit(1)
