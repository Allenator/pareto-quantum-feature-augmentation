"""Run the QuantumReservoir locally (lightning.qubit, exact) on synthetic data.

Uses the same circuit as quantum_reservoir.py and the same parameters as
run_synthetic_hw_packed.py for direct comparison with hardware results.

Usage:
    uv run python scripts/run_synthetic_local.py                     # default 100+50
    uv run python scripts/run_synthetic_local.py 500 250             # custom sizes
    uv run python scripts/run_synthetic_local.py 100 50 500 250 1000 500  # multiple runs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.quantum_reservoir import QuantumReservoir
from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

# ── Configuration (matches run_synthetic_hw_packed.py) ───────────────────
N_QUBITS = 4
N_ENSEMBLE = 3       # = N_RESERVOIRS in hw scripts
N_LAYERS = 3
SEED = 42
CLIP_RANGE = 5.0

FEATURES_DIR = Path("features/synthetic_hw")


def run(n_train: int, n_test: int):
    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)

    # Run QuantumReservoir (exact simulator, no shot noise)
    t0 = time.perf_counter()
    qr = QuantumReservoir(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_ensemble=N_ENSEMBLE,
        seed=SEED,
    )
    X_new_train = qr.transform(X_train_s)
    X_new_test = qr.transform(X_test_s)
    elapsed = time.perf_counter() - t0

    # Prepend original features (matches hw packed script output format)
    train_features = np.hstack([X_train_s, X_new_train])
    test_features = np.hstack([X_test_s, X_new_test])

    # Save
    aug_name = f"reservoir_{N_ENSEMBLE}x{N_LAYERS}_Z"
    feat_dir = FEATURES_DIR / "lightning" / f"seed_{SEED}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{aug_name}_n{n_train}+{n_test}_exact.npz"
    np.savez_compressed(feat_path, train=train_features, test=test_features)

    # Ridge regression
    model = RidgeModel()
    pred = model.fit_predict(train_features, y_train, test_features)
    mse = float(np.mean((pred.y_pred - y_test) ** 2))

    # Report
    print(f"{'=' * 60}")
    print("LOCAL SIMULATION RESULTS (lightning.qubit, exact)")
    print(f"{'=' * 60}")
    print(f"  Augmenter:       {aug_name}")
    print(f"  Ensemble:        {N_ENSEMBLE} circuits × {N_LAYERS} layers")
    print(f"  Train features:  {train_features.shape}")
    print(f"  Test features:   {test_features.shape}")
    print(f"  Wall clock:      {elapsed:.2f} s")
    print(f"  Saved features:  {feat_path}")
    print(f"  Ridge test MSE:  {mse:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Default: 100+50
        run(100, 50)
    else:
        # Parse pairs: n_train n_test [n_train n_test ...]
        if len(args) % 2 != 0:
            print("Usage: run_synthetic_local.py [n_train n_test ...]")
            sys.exit(1)
        for i in range(0, len(args), 2):
            run(int(args[i]), int(args[i + 1]))
