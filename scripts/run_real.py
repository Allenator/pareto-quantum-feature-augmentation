"""Run the real financial data walk-forward backtest experiment."""

import sys
from concurrent.futures import ProcessPoolExecutor
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

SEEDS = [42, 123, 456, 789, 1024]


def _with_seed(augmenters: list, seed: int) -> list:
    """Return a copy of augmenter configs with the given seed injected."""
    out = []
    for cfg in augmenters:
        if cfg.params and ("seed" in cfg.params or cfg.kind == "quantum_fixed"):
            params = {**cfg.params, "seed": seed}
            out.append(AugmenterConfig(cfg.name, cfg.kind, params))
        elif cfg.kind == "classical" and cfg.params and "n_components" in cfg.params:
            # RFF augmenters: use seed as random_state
            params = {**cfg.params, "random_state": seed}
            out.append(AugmenterConfig(cfg.name, cfg.kind, params))
        else:
            out.append(cfg)
    return out


def run_quick():
    """Quick test: 3 tickers, monthly eval steps, minimal augmenters, 1 seed."""
    quick_augs = [
        AugmenterConfig("identity", "classical"),
        AugmenterConfig("poly_deg2", "classical"),
        AugmenterConfig("qunified_z_8q_3L_3ens", "quantum_fixed",
            {**_UNIFIED_BASE, "n_qubits": 8, "observables": "Z",
             "n_layers": 3, "n_ensemble": 3}),
    ]
    config = ExperimentConfig(
        data=RealDataConfig(
            tickers=("AAPL", "MSFT", "NVDA"),
            start_date="2022-01-01",
            end_date="2025-12-31",
        ),
        backtest=BacktestConfig(step_days=21),
        augmenters=quick_augs,
        models=[ModelConfig("ridge")],
        run_id="quick",
        corr_augmenter=_corr_quantum(3),
    )
    runner = BacktestRunner(config)
    return runner.run()


def _run_single_seed(args):
    """Run a single seed's backtest. Top-level function for pickling."""
    seed, step_days, run_prefix = args
    corr_cfg = _corr_quantum(10)
    corr_cfg["seed"] = seed
    config = ExperimentConfig(
        data=RealDataConfig(),
        backtest=BacktestConfig(step_days=step_days),
        augmenters=_with_seed(CLASSICAL + QUANTUM, seed),
        models=MODELS,
        run_id=f"{run_prefix}_s{seed}",
        corr_augmenter=corr_cfg,
    )
    runner = BacktestRunner(config)
    return runner.run()


def _run_seeds_parallel(step_days, run_prefix):
    """Run all seeds in parallel processes."""
    args = [(seed, step_days, run_prefix) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        list(pool.map(_run_single_seed, args))


def run_monthly():
    """All augmenters, 10 tickers, monthly eval steps, 5 seeds in parallel."""
    _run_seeds_parallel(step_days=21, run_prefix="monthly")


def run_full():
    """Full experiment: 10 tickers, daily eval, all augmenters, 5 seeds in parallel."""
    _run_seeds_parallel(step_days=1, run_prefix="full")


def _ablation_config(n_tickers, step_days, prefix, use_regime, use_corr, seed):
    """Build ExperimentConfig for ablation study."""
    tickers = ("AAPL", "MSFT", "NVDA") if n_tickers == 3 else RealDataConfig().tickers
    tag = ("regime" if use_regime else "noregime") + ("_corr" if use_corr else "_nocorr")
    n_feat = 17 if use_regime else 14
    base = dict(n_features=n_feat, encoding="angle", connectivity="circular",
                cnot_mixing=True, random_rot=True)
    augs = [
        AugmenterConfig("identity", "classical"),
        AugmenterConfig("qunified_z_8q_3L_3ens", "quantum_fixed",
            {**base, "n_qubits": 8, "observables": "Z", "n_layers": 3, "n_ensemble": 3}),
    ]
    corr_cfg = None
    if use_corr:
        corr_cfg = _corr_quantum(n_tickers)
        corr_cfg["seed"] = seed
    return ExperimentConfig(
        data=RealDataConfig(tickers=tickers),
        backtest=BacktestConfig(step_days=step_days),
        augmenters=_with_seed(augs, seed),
        models=[ModelConfig("ridge")],
        run_id=f"{prefix}_{tag}_s{seed}",
        use_regime_features=use_regime,
        corr_augmenter=corr_cfg,
    )


def _run_single_ablation(args):
    """Run a single ablation config. Top-level function for pickling."""
    n_tickers, step_days, prefix, use_regime, use_corr, seed = args
    config = _ablation_config(n_tickers, step_days, prefix, use_regime, use_corr, seed)
    runner = BacktestRunner(config)
    return runner.run()


def _run_ablation_grid(n_tickers, step_days, prefix):
    """Run 2×2 ablation × 5 seeds in parallel."""
    args = [
        (n_tickers, step_days, prefix, use_regime, use_corr, seed)
        for seed in SEEDS
        for use_regime in [False, True]
        for use_corr in [False, True]
    ]
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        list(pool.map(_run_single_ablation, args))


def run_ablation():
    """2×2 ablation × 5 seeds: 3 tickers, monthly (quick diagnostic)."""
    _run_ablation_grid(n_tickers=3, step_days=21, prefix="ablation")


def run_ablation_monthly():
    """2×2 ablation × 5 seeds: 10 tickers, monthly eval."""
    _run_ablation_grid(n_tickers=10, step_days=21, prefix="ablation_monthly")


def run_ablation_full():
    """2×2 ablation × 5 seeds: 10 tickers, daily eval."""
    _run_ablation_grid(n_tickers=10, step_days=1, prefix="ablation_full")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    modes = {
        "quick": run_quick,
        "monthly": run_monthly,
        "full": run_full,
        "ablation": run_ablation,
        "ablation-monthly": run_ablation_monthly,
        "ablation-full": run_ablation_full,
    }
    if mode in modes:
        modes[mode]()
    else:
        print(f"Usage: python scripts/run_real.py [{' | '.join(modes)}]")
        sys.exit(1)
