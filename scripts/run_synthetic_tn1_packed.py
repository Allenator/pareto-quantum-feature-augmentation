"""Run reservoir augmenter on AWS Braket TN1 simulator using native circuits with qubit packing.

Same packed circuit approach as run_synthetic_hw_packed.py but targeting TN1
(tensor-network simulator with shot sampling, up to 50 qubits). This provides
a noise-free baseline with shot noise only, for comparison against QPU results.

Usage:
    uv run python scripts/run_synthetic_tn1_packed.py
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

BACKEND = "tn1"
DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"
TOTAL_QUBITS = 50    # TN1 supports up to 50 qubits

N_TRAIN = 100
N_TEST = 50
CLIP_RANGE = 5.0
FEATURES_DIR = Path("features/synthetic_hw")


# ── 4. Native Braket circuit builder with qubit packing ──────────────────
# (identical to run_synthetic_hw_packed.py)

def build_packed_circuit(
    samples: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
) -> Circuit:
    """Build a single circuit packing multiple samples on disjoint qubit subsets.

    PennyLane Rot(φ,θ,ω) = RZ(φ) → RY(θ) → RZ(ω) in Braket gate order.
    """
    n_pack = len(samples)
    circ = Circuit()

    for slot in range(n_pack):
        offset = slot * n_qubits
        x = samples[slot]

        # 1. Data encoding: RY(x[i])
        for i in range(n_qubits):
            circ.ry(offset + i, float(x[i]))

        # 2. Layers: Rot(φ,θ,ω) + linear CNOTs
        for layer in range(weights.shape[0]):
            for i in range(n_qubits):
                phi, theta, omega = weights[layer, i]
                circ.rz(offset + i, float(phi))
                circ.ry(offset + i, float(theta))
                circ.rz(offset + i, float(omega))
            for i in range(n_qubits - 1):
                circ.cnot(offset + i, offset + i + 1)

    # 3. Measurements: ⟨Z_i⟩ + ⟨Z_i Z_j⟩ for each slot
    for slot in range(n_pack):
        offset = slot * n_qubits
        for i in range(n_qubits):
            circ.expectation(Observable.Z(), target=offset + i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circ.expectation(
                    Observable.Z() @ Observable.Z(),
                    target=[offset + i, offset + j],
                )

    return circ


def n_features_per_sample(n_qubits: int) -> int:
    return n_qubits + n_qubits * (n_qubits - 1) // 2


def run_packed_reservoir(
    X: np.ndarray,
    reservoir_weights: list[np.ndarray],
    n_qubits: int,
    pack_factor: int,
    device: AwsDevice,
    shots: int,
) -> np.ndarray:
    """Run all reservoir circuits with qubit packing."""
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

            padded = np.array([_pad_features(x, n_qubits) for x in batch_X])
            circ = build_packed_circuit(padded, weights, n_qubits)

            print(f"\r  Reservoir {r_idx+1}/{len(reservoir_weights)}  "
                  f"batch {batch_idx+1}/{n_batches}  "
                  f"({n_in_batch} samples packed on "
                  f"{n_in_batch * n_qubits} qubits)", end="", flush=True)

            task = device.run(circ, shots=shots)
            result = task.result()

            values = result.values
            for slot in range(n_in_batch):
                slot_values = values[slot * n_feat : (slot + 1) * n_feat]
                sample_features[batch_start + slot] = slot_values

        all_reservoir_features.append(sample_features)

    print()
    return np.hstack(all_reservoir_features)


# ── 5. Main ──────────────────────────────────────────────────────────────
def main():
    pack_factor = TOTAL_QUBITS // N_QUBITS  # 34 // 4 = 8

    # Generate random reservoir weights (same seeds as all other scripts)
    reservoir_weights = []
    for r in range(N_RESERVOIRS):
        rng = np.random.default_rng(SEED + r)
        reservoir_weights.append(rng.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3)))

    # Load data
    dgp = DGPConfig(n_train=N_TRAIN, n_test=N_TEST, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)

    X_all = np.vstack([X_train_s, X_test_s])
    total_samples = len(X_all)

    n_batches_per_reservoir = math.ceil(total_samples / pack_factor)
    n_tasks = n_batches_per_reservoir * N_RESERVOIRS

    # TN1 pricing: $0.275 per minute of simulation
    print(f"\n{'=' * 60}")
    print("COST ESTIMATION (TN1 packed)")
    print(f"{'=' * 60}")
    print(f"  Backend:            TN1 (tensor-network simulator)")
    print(f"  Total qubits:       {TOTAL_QUBITS}")
    print(f"  Pack factor:        {pack_factor} samples/task")
    print(f"  Samples:            {total_samples} (train + test)")
    print(f"  Reservoirs:         {N_RESERVOIRS}")
    print(f"  Tasks:              {n_tasks}")
    print(f"  Shots per task:     {SHOTS:,}")
    print(f"  Pricing:            $0.275/min simulation time")
    print(f"{'=' * 60}\n")

    resp = input("Proceed with execution? [y/N] ")
    if resp.strip().lower() != "y":
        print("Aborted.")
        return

    device = AwsDevice(DEVICE_ARN)
    print(f"Device: {device.name}")

    cost_tracker = Tracker().start()
    t0 = time.perf_counter()

    print("Running packed circuits on TN1...")
    X_new_all = run_packed_reservoir(
        X_all, reservoir_weights, N_QUBITS, pack_factor, device, SHOTS,
    )

    elapsed = time.perf_counter() - t0
    cost_tracker.stop()

    # Split back into train / test and prepend original features
    n_train = len(X_train_s)
    train_features = np.hstack([X_train_s, X_new_all[:n_train]])
    test_features = np.hstack([X_test_s, X_new_all[n_train:]])

    # Save
    aug_name = f"reservoir_{N_RESERVOIRS}x{N_LAYERS}_Z+ZZ"
    feat_dir = FEATURES_DIR / BACKEND / f"seed_{SEED}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{aug_name}_n{N_TRAIN}+{N_TEST}_s{SHOTS}_packed.npz"
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
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
