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

N_FEATURES = 17  # 14 per-stock + 3 regime (FEATURE_COLS + REGIME_COLS)

# Common kwargs for the winning synthetic design axes
_UNIFIED_BASE = dict(
    n_features=N_FEATURES, encoding="angle", connectivity="circular",
    cnot_mixing=True, random_rot=True,
)

# Correlation quantum augmenter config (Approach 3: quantum encoding of corr matrix)
# n_features = n_tickers * (n_tickers - 1) // 2 (upper triangle of corr matrix)
def _corr_quantum(n_tickers: int) -> dict:
    n_pairs = n_tickers * (n_tickers - 1) // 2
    return dict(
        n_features=n_pairs, n_qubits=min(8, n_pairs), encoding="angle",
        connectivity="circular", cnot_mixing=True, random_rot=True,
        observables="Z", n_layers=3, n_ensemble=3, qubit_mapping="modular", seed=42,
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
        corr_augmenter=_corr_quantum(3),
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
        corr_augmenter=_corr_quantum(10),
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
        corr_augmenter=_corr_quantum(10),
    )
    runner = BacktestRunner(config)
    return runner.run()


def _ablation_config(n_tickers, use_regime, use_corr):
    """Build ExperimentConfig for ablation study."""
    tickers = ("AAPL", "MSFT", "NVDA") if n_tickers == 3 else RealDataConfig().tickers
    tag = ("regime" if use_regime else "noregime") + ("_corr" if use_corr else "_nocorr")
    # n_features must match: 17 with regime, 14 without
    n_feat = 17 if use_regime else 14
    base = dict(n_features=n_feat, encoding="angle", connectivity="circular",
                cnot_mixing=True, random_rot=True)
    return ExperimentConfig(
        data=RealDataConfig(tickers=tickers),
        backtest=BacktestConfig(step_days=21),
        augmenters=[
            AugmenterConfig("identity", "classical"),
            AugmenterConfig("qunified_z_8q_3L_3ens", "quantum_fixed",
                {**base, "n_qubits": 8, "observables": "Z", "n_layers": 3, "n_ensemble": 3}),
        ],
        models=[ModelConfig("ridge")],
        run_id=f"ablation_{tag}",
        use_regime_features=use_regime,
        corr_augmenter=_corr_quantum(n_tickers) if use_corr else None,
    )


def run_ablation():
    """2x2 ablation: regime features × correlation quantum encoding."""
    for use_regime in [False, True]:
        for use_corr in [False, True]:
            config = _ablation_config(3, use_regime, use_corr)
            print(f"\n{'='*60}")
            print(f"Ablation: regime={use_regime}, corr_quantum={use_corr}")
            print(f"{'='*60}")
            runner = BacktestRunner(config)
            runner.run()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    if mode == "quick":
        run_quick()
    elif mode == "monthly":
        run_monthly()
    elif mode == "full":
        run_full()
    elif mode == "ablation":
        run_ablation()
    else:
        print(f"Usage: python scripts/run_real.py [quick|monthly|full|ablation]")
        sys.exit(1)
