"""Singleton-mode hardware run: one 4-qubit reservoir circuit per Braket task.

Identical to run_synthetic_hw_packed.py in every respect except the qubit
packing factor. Where the packed script fits 20 independent 4-qubit circuit
slots into a single 80-qubit Rigetti Ankaa-3 task (to amortise the per-task
fee), this script submits each sample as its own small, clean task that uses
only 4 qubits. No idle qubits, no parallel-gate crosstalk between slots.

Purpose
-------
Isolate hardware noise caused by running on ~all of Ankaa-3's qubits at once.
The packed run (at ~80q density) can be directly compared to this singleton
run (at 4q density) on *exactly* the same input data — same seed, same DGP
subset, same reservoir weights, same observables, same shot count — so the
only difference is the number of active qubits per task.

Cost (Rigetti Ankaa-3 on-demand)
--------------------------------
  $0.30 per task + $0.0009 per shot
  n_train = 100, n_test = 50 → 150 samples
  3 reservoirs × 150 samples × 1 sample/task = 450 tasks
  450 tasks × ($0.30 + 1000 × $0.0009) = $540 total (approx)

If your intent is to compare against the existing
`reservoir_3x3_Z_n100+50_s1000_packed.npz`, the inputs and reservoir weights
will match byte-for-byte and the output `.npz` can be fed into
`scripts/compare_features.py` alongside the packed and exact files.

Usage
-----
    uv run python scripts/run_synthetic_hw_singleton.py             # default 100+50
    uv run python scripts/run_synthetic_hw_singleton.py 50 25       # custom size
"""

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 1. AWS credentials (loaded from .env.aws, gitignored) ────────────────
_env_file = Path(__file__).resolve().parent.parent / ".env.aws"
if not _env_file.exists():
    print(f"ERROR: AWS credentials file not found: {_env_file}")
    print("Create it with contents like:")
    print('  AWS_DEFAULT_REGION="us-east-1"')
    print('  AWS_ACCESS_KEY_ID="..."')
    print('  AWS_SECRET_ACCESS_KEY="..."')
    print('  AWS_SESSION_TOKEN="..."')
    sys.exit(1)

with open(_env_file) as _f:
    for _line in _f:
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if _line.startswith("export "):
            _line = _line[len("export "):]
        if "=" in _line:
            _key, _val = _line.split("=", 1)
            os.environ[_key.strip()] = _val.strip().strip('"')

# ── 2. Imports (after credentials are set) ───────────────────────────────
import numpy as np
from braket.aws import AwsDevice
from braket.tracking import Tracker
from sklearn.preprocessing import StandardScaler

from scripts.run_synthetic_hw_packed import build_packed_circuit, n_features_per_sample
from src.synthetic.augmenters.quantum_fixed import _pad_features
from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

# ── 3. Experiment configuration ─────────────────────────────────────────
N_QUBITS = 4
N_RESERVOIRS = 3
N_LAYERS = 3
SHOTS = 1000
SEED = 42
PACK_FACTOR = 1          # ← the key difference from run_synthetic_hw_packed.py

# Backend (hardcoded: singleton crosstalk test is Rigetti-specific)
BACKEND = "rigetti"
DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
PER_TASK = 0.30
PER_SHOT = 0.0009

# Default dataset sizes — 1/10 of the 1000+500 packed run so cost ≈ 2× the
# packed 1000+500 run. Override with CLI args.
N_TRAIN = 100
N_TEST = 50
CLIP_RANGE = 5.0
FEATURES_DIR = Path("features/synthetic_hw")


def estimate_cost(n_samples: int, n_reservoirs: int, shots: int,
                  pack_factor: int) -> int:
    """Print cost estimate and return task count."""
    n_batches = math.ceil(n_samples / pack_factor)
    n_tasks = n_batches * n_reservoirs
    total_shots = n_tasks * shots
    task_cost = n_tasks * PER_TASK
    shot_cost = total_shots * PER_SHOT
    print(f"\n{'=' * 60}")
    print("COST ESTIMATION (singleton — pack_factor = 1)")
    print(f"{'=' * 60}")
    print(f"  Backend:          Rigetti Ankaa-3")
    print(f"  Active qubits/task: {N_QUBITS}  (of 82)")
    print(f"  Pack factor:      {pack_factor} sample(s)/task")
    print(f"  Samples:          {n_samples} (train + test)")
    print(f"  Reservoirs:       {n_reservoirs}")
    print(f"  Tasks:            {n_tasks} ({n_batches}/reservoir × {n_reservoirs})")
    print(f"  Shots per task:   {shots:,}")
    print(f"  Total shots:      {total_shots:,}")
    print(f"  Task cost:        ${task_cost:.2f}  ({n_tasks} × ${PER_TASK})")
    print(f"  Shot cost:        ${shot_cost:.2f}  ({total_shots:,} × ${PER_SHOT})")
    print(f"  Est. total cost:  ${task_cost + shot_cost:.2f} USD")
    print(f"{'=' * 60}\n")
    return n_tasks


def run_singleton_reservoir(X: np.ndarray,
                             reservoir_weights: list,
                             n_qubits: int,
                             device: AwsDevice,
                             shots: int) -> np.ndarray:
    """Submit one task per (sample, reservoir) combination.

    Uses build_packed_circuit with n_pack = 1 so the circuit fragment is
    identical to a single slot of the packed run — no code duplication.
    """
    n_samples = len(X)
    n_feat = n_features_per_sample(n_qubits)
    all_reservoir_features = []

    for r_idx, weights in enumerate(reservoir_weights):
        sample_features = np.zeros((n_samples, n_feat))
        for s_idx, x in enumerate(X):
            padded = np.array([_pad_features(x, n_qubits)])  # shape (1, n_qubits)
            circ = build_packed_circuit(padded, weights, n_qubits)

            print(f"\r  Reservoir {r_idx + 1}/{len(reservoir_weights)}  "
                  f"sample {s_idx + 1}/{n_samples}  "
                  f"(4 qubits active)", end="", flush=True)

            task = device.run(circ, shots=shots)
            result = task.result()
            sample_features[s_idx] = result.values[:n_feat]

        all_reservoir_features.append(sample_features)

    print()
    return np.hstack(all_reservoir_features)


def run(n_train: int, n_test: int):
    # Generate reservoir weights — MUST match the packed script's seed
    # convention so weights are identical between packed and singleton runs.
    reservoir_weights = []
    for r in range(N_RESERVOIRS):
        rng = np.random.default_rng(SEED + r)
        reservoir_weights.append(
            rng.uniform(0, 2 * np.pi, (N_LAYERS + 1, N_QUBITS, 3)))

    # Load data — re-uses the same cached parquet as all prior runs, so the
    # first 4 columns of the saved features will match the packed counterpart
    # byte-for-byte.
    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)
    X_all = np.vstack([X_train_s, X_test_s])
    total_samples = len(X_all)

    estimate_cost(total_samples, N_RESERVOIRS, SHOTS, PACK_FACTOR)

    resp = input("Proceed with execution? [y/N] ")
    if resp.strip().lower() != "y":
        print("Aborted.")
        return

    device = AwsDevice(DEVICE_ARN)
    print(f"Device: {device.name} ({DEVICE_ARN})")
    print(f"Status: {device.status}\n")

    cost_tracker = Tracker().start()
    t0 = time.perf_counter()

    print("Running singleton (4-qubit) circuits on QPU...")
    X_new_all = run_singleton_reservoir(
        X_all, reservoir_weights, N_QUBITS, device, SHOTS,
    )

    elapsed = time.perf_counter() - t0
    cost_tracker.stop()

    # Split back, prepend original features (schema matches packed output)
    n_tr = len(X_train_s)
    train_features = np.hstack([X_train_s, X_new_all[:n_tr]])
    test_features = np.hstack([X_test_s, X_new_all[n_tr:]])

    aug_name = f"reservoir_{N_RESERVOIRS}x{N_LAYERS}_Z"
    feat_dir = FEATURES_DIR / BACKEND / f"seed_{SEED}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{aug_name}_n{n_train}+{n_test}_s{SHOTS}_singleton.npz"
    np.savez_compressed(feat_path, train=train_features, test=test_features)

    print(f"\n{'=' * 60}")
    print("RESULTS (singleton)")
    print(f"{'=' * 60}")
    print(f"  Augmenter:       {aug_name}")
    print(f"  Train features:  {train_features.shape}")
    print(f"  Test features:   {test_features.shape}")
    print(f"  Wall clock:      {elapsed:.1f} s")
    print(f"  Pack factor:     {PACK_FACTOR} (4-qubit tasks)")
    print(f"  Saved features:  {feat_path}")

    # Quick Ridge sanity check
    pred = RidgeModel().fit_predict(train_features, y_train, test_features)
    mse = float(np.mean((pred.y_pred - y_test) ** 2))
    print(f"  Ridge test MSE:  {mse:.4f}")

    sim_cost = cost_tracker.simulator_tasks_cost()
    qpu_cost = cost_tracker.qpu_tasks_cost()
    print(f"\n{'=' * 60}")
    print("ACTUAL COST")
    print(f"{'=' * 60}")
    print(f"  Simulator cost:  ${sim_cost:.4f} USD")
    print(f"  QPU cost:        ${qpu_cost:.4f} USD")
    print(f"  Total cost:      ${sim_cost + qpu_cost:.4f} USD")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        run(N_TRAIN, N_TEST)
    else:
        if len(args) % 2 != 0:
            print("Usage: run_synthetic_hw_singleton.py [n_train n_test ...]")
            sys.exit(1)
        for i in range(0, len(args), 2):
            run(int(args[i]), int(args[i + 1]))
