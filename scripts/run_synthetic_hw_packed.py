"""Run reservoir augmenter on AWS Braket QPU using native circuits with qubit packing.

Packs multiple independent 4-qubit reservoir circuits onto disjoint qubit
subsets of a larger QPU (e.g., 20 qubits → 5 samples/task, 84 qubits →
20 samples/task), drastically reducing per-task cost.

Circuit structure (matches scripts/quantum_reservoir.py):
  1. RY(x[i]) on each qubit                          (data encoding)
  2. n_layers × [Rot(φ,θ,ω) + linear CNOTs]          (random fixed weights)
  3. Final Rot(φ,θ,ω) on each qubit                   (extra rotation before measurement)
  4. Measure ⟨Z_i⟩                                    (single-qubit Z only)

Usage:
    uv run python scripts/run_synthetic_hw_packed.py                     # default 100+50
    uv run python scripts/run_synthetic_hw_packed.py 500 250             # custom sizes
    uv run python scripts/run_synthetic_hw_packed.py 100 50 500 250      # multiple runs
    BRAKET_BACKEND=iqm uv run python scripts/run_synthetic_hw_packed.py
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
from braket.circuits import Circuit, Observable
from braket.tracking import Tracker
from sklearn.preprocessing import StandardScaler

from src.synthetic.augmenters.quantum_fixed import _pad_features
from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

# ── 3. Experiment configuration ─────────────────────────────────────────
N_QUBITS = 4
N_RESERVOIRS = 3     # Matches documented reservoir_3x3_zz config
N_LAYERS = 3
SHOTS = 1000
SEED = 42

# Backend selection
BACKEND = os.environ.get("BRAKET_BACKEND", "rigetti")

DEVICE_MAP = {
    "rigetti": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
    "iqm":     "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    "sv1":     "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
}

# QPU qubit counts (for packing calculation)
DEVICE_QUBITS = {
    "rigetti": 80,
    "iqm":     20,
    "sv1":     34,
}

# Pricing: (per_task, per_shot)
DEVICE_PRICING = {
    "rigetti": (0.30, 0.0009),
    "iqm":     (0.30, 0.00145),
    "sv1":     (0.0, 0.0),
}

N_TRAIN = 100
N_TEST = 50
CLIP_RANGE = 5.0
FEATURES_DIR = Path("features/synthetic_hw")


# ── 4. Native Braket circuit builder with qubit packing ──────────────────
def build_packed_circuit(
    samples: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
) -> Circuit:
    """Build a single circuit packing multiple samples on disjoint qubit subsets.

    Matches the circuit in scripts/quantum_reservoir.py:
        RY(x_i) → [Rot + CNOT_linear] × n_layers → Rot → ⟨Z_i⟩

    Each sample occupies [slot*n_qubits, (slot+1)*n_qubits) wires.
    PennyLane Rot(φ,θ,ω) = RZ(φ) → RY(θ) → RZ(ω) in Braket gate order.

    Args:
        samples: (n_pack, n_qubits) array of padded input features
        weights: (n_layers + 1, n_qubits, 3) reservoir weights
                 (last slice is the final Rot before measurement)
        n_qubits: qubits per sample
    """
    n_pack = len(samples)
    n_layers = weights.shape[0] - 1  # last slice is the final Rot
    circ = Circuit()

    for slot in range(n_pack):
        offset = slot * n_qubits
        x = samples[slot]

        # 1. Data encoding: RY(x[i])
        for i in range(n_qubits):
            circ.ry(offset + i, float(x[i]))

        # 2. Mixing layers: Rot(φ,θ,ω) + linear CNOTs
        for layer in range(n_layers):
            for i in range(n_qubits):
                phi, theta, omega = weights[layer, i]
                circ.rz(offset + i, float(phi))
                circ.ry(offset + i, float(theta))
                circ.rz(offset + i, float(omega))
            for i in range(n_qubits - 1):
                circ.cnot(offset + i, offset + i + 1)

        # 3. Final Rot before measurement
        for i in range(n_qubits):
            phi, theta, omega = weights[n_layers, i]
            circ.rz(offset + i, float(phi))
            circ.ry(offset + i, float(theta))
            circ.rz(offset + i, float(omega))

    # 4. Measurements: ⟨Z_i⟩ for each slot
    for slot in range(n_pack):
        offset = slot * n_qubits
        for i in range(n_qubits):
            circ.expectation(Observable.Z(), target=offset + i)

    return circ


def n_features_per_sample(n_qubits: int) -> int:
    """Number of Z features per sample."""
    return n_qubits


def run_packed_reservoir(
    X: np.ndarray,
    reservoir_weights: list[np.ndarray],
    n_qubits: int,
    pack_factor: int,
    device: AwsDevice,
    shots: int,
) -> np.ndarray:
    """Run all reservoir circuits with qubit packing.

    Returns: (n_samples, n_reservoirs * n_features_per_sample) array.
    """
    n_samples = len(X)
    n_feat = n_features_per_sample(n_qubits)
    all_reservoir_features = []

    for r_idx, weights in enumerate(reservoir_weights):
        sample_features = np.zeros((n_samples, n_feat))

        n_batches = math.ceil(n_samples / pack_factor)
        for batch_idx in range(n_batches):
            batch_start = batch_idx * pack_factor
            batch_end = min(batch_start + pack_factor, n_samples)
            batch_X = X[batch_start:batch_end]
            n_in_batch = len(batch_X)

            # Pad features for each sample
            padded = np.array([_pad_features(x, n_qubits) for x in batch_X])

            # Build packed circuit
            circ = build_packed_circuit(padded, weights, n_qubits)

            print(f"\r  Reservoir {r_idx+1}/{len(reservoir_weights)}  "
                  f"batch {batch_idx+1}/{n_batches}  "
                  f"({n_in_batch} samples packed on "
                  f"{n_in_batch * n_qubits} qubits)", end="", flush=True)

            task = device.run(circ, shots=shots)
            result = task.result()

            # Extract expectation values — ordered by measurement declaration
            values = result.values
            for slot in range(n_in_batch):
                slot_values = values[slot * n_feat : (slot + 1) * n_feat]
                sample_features[batch_start + slot] = slot_values

        all_reservoir_features.append(sample_features)

    print()
    return np.hstack(all_reservoir_features)


# ── 5. Cost estimation ──────────────────────────────────────────────────
def estimate_cost(n_samples, n_reservoirs, shots, backend, pack_factor):
    """Print cost estimate with qubit packing."""
    n_batches_per_reservoir = math.ceil(n_samples / pack_factor)
    n_tasks = n_batches_per_reservoir * n_reservoirs
    total_shots = n_tasks * shots

    print(f"\n{'=' * 60}")
    print("COST ESTIMATION (with qubit packing)")
    print(f"{'=' * 60}")
    print(f"  Backend:            {backend}")
    print(f"  QPU qubits:         {DEVICE_QUBITS[backend]}")
    print(f"  Qubits per sample:  {N_QUBITS}")
    print(f"  Pack factor:        {pack_factor} samples/task")
    print(f"  Samples:            {n_samples} (train + test)")
    print(f"  Reservoirs:         {n_reservoirs}")
    print(f"  Tasks:              {n_tasks} ({n_batches_per_reservoir}/reservoir × {n_reservoirs})")
    print(f"  Shots per task:     {shots:,}")
    print(f"  Total shots:        {total_shots:,}")

    if backend in DEVICE_PRICING:
        per_task, per_shot = DEVICE_PRICING[backend]
        task_cost = n_tasks * per_task
        shot_cost = total_shots * per_shot
        est_cost = task_cost + shot_cost
        print(f"  Task cost:          ${task_cost:.2f}  ({n_tasks} × ${per_task})")
        print(f"  Shot cost:          ${shot_cost:.2f}  ({total_shots:,} × ${per_shot})")
        print(f"  Est. total cost:    ${est_cost:.2f} USD")

    print(f"{'=' * 60}\n")


# ── 6. Main ──────────────────────────────────────────────────────────────
def run(n_train: int, n_test: int):
    if BACKEND not in DEVICE_MAP:
        print(f"Unknown backend: {BACKEND!r}. Choose from: {list(DEVICE_MAP.keys())}")
        sys.exit(1)

    device_arn = DEVICE_MAP[BACKEND]
    total_qubits = DEVICE_QUBITS[BACKEND]
    pack_factor = total_qubits // N_QUBITS

    # Generate random reservoir weights (matches quantum_reservoir.py seed convention)
    reservoir_weights = []
    for r in range(N_RESERVOIRS):
        rng = np.random.default_rng(SEED + r)
        # n_layers + 1: last slice is the final Rot before measurement
        reservoir_weights.append(rng.uniform(0, 2 * np.pi, (N_LAYERS + 1, N_QUBITS, 3)))

    # Load data
    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)

    # Combine train + test for one pass through hardware
    X_all = np.vstack([X_train_s, X_test_s])
    total_samples = len(X_all)

    estimate_cost(total_samples, N_RESERVOIRS, SHOTS, BACKEND, pack_factor)

    resp = input("Proceed with execution? [y/N] ")
    if resp.strip().lower() != "y":
        print("Aborted.")
        return

    # Connect to device
    device = AwsDevice(device_arn)
    print(f"Device: {device.name} ({device_arn})")
    print(f"Status: {device.status}\n")

    cost_tracker = Tracker().start()
    t0 = time.perf_counter()

    # Run all samples through packed reservoir circuits
    print("Running packed circuits on QPU...")
    X_new_all = run_packed_reservoir(
        X_all, reservoir_weights, N_QUBITS, pack_factor, device, SHOTS,
    )

    elapsed = time.perf_counter() - t0
    cost_tracker.stop()

    # Split back into train / test and prepend original features
    n_tr = len(X_train_s)
    train_features = np.hstack([X_train_s, X_new_all[:n_tr]])
    test_features = np.hstack([X_test_s, X_new_all[n_tr:]])

    # Save
    aug_name = f"reservoir_{N_RESERVOIRS}x{N_LAYERS}_Z"
    feat_dir = FEATURES_DIR / BACKEND / f"seed_{SEED}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{aug_name}_n{n_train}+{n_test}_s{SHOTS}_packed.npz"
    np.savez_compressed(feat_path, train=train_features, test=test_features)

    # Results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"  Augmenter:       {aug_name}")
    print(f"  Train features:  {train_features.shape}")
    print(f"  Test features:   {test_features.shape}")
    print(f"  Wall clock:      {elapsed:.1f} s")
    print(f"  Pack factor:     {pack_factor} samples/task")
    print(f"  Saved features:  {feat_path}")

    # Ridge regression
    model = RidgeModel()
    pred = model.fit_predict(train_features, y_train, test_features)
    mse = float(np.mean((pred.y_pred - y_test) ** 2))
    print(f"  Ridge test MSE:  {mse:.4f}")

    # Actual cost
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
            print("Usage: run_synthetic_hw_packed.py [n_train n_test ...]")
            sys.exit(1)
        for i in range(0, len(args), 2):
            run(int(args[i]), int(args[i + 1]))
