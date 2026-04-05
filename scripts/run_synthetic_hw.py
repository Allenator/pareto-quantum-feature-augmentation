"""Run ONE quantum augmenter on an AWS Braket device with cost tracking.

Usage:
    uv run python scripts/run_synthetic_hw.py          # default: SV1 simulator
    BRAKET_BACKEND=local uv run python scripts/run_synthetic_hw.py   # local Braket sim
"""

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
import pennylane as qml
from braket.devices import Devices
from braket.tracking import Tracker
from sklearn.preprocessing import StandardScaler

from src.synthetic.augmenters.quantum_fixed import ReservoirAugmenter
from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.evaluation.metrics import compute_metrics
from src.synthetic.models.linear import RidgeModel

# ── 3. Experiment configuration ─────────────────────────────────────────
N_QUBITS = 4
N_RESERVOIRS = 3     # Matches documented reservoir_3x3_zz config
N_LAYERS = 3
OBSERVABLES = "Z+ZZ"  # All commute → 1 measurement group (vs 3 for XYZ+ZZ)
SHOTS = 1000

# Backend selection
BACKEND = os.environ.get("BRAKET_BACKEND", "rigetti")

# (pl_device_name, device_arn)
DEVICE_MAP = {
    "sv1":     ("braket.aws.qubit", Devices.Amazon.SV1),
    "dm1":     ("braket.aws.qubit", Devices.Amazon.DM1),
    "local":   ("braket.local.qubit", None),
    "iqm":     ("braket.aws.qubit", "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"),
    "rigetti": ("braket.aws.qubit", "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"),
}

N_TRAIN = 100
N_TEST = 50
SEED = 42
CLIP_RANGE = 5.0

# Output directory for saved augmented features (NPZ)
FEATURES_DIR = Path("features/synthetic_hw")


# ── 4. Cost estimation ──────────────────────────────────────────────────
def estimate_cost(n_samples: int, n_reservoirs: int, shots: int, backend: str):
    """Print a rough cost estimate BEFORE execution."""
    # Each reservoir processes all samples in ONE batched call (parameter
    # broadcasting). But with shot-based devices, non-commuting observables
    # (X, Y, Z on the same qubit) are split into ~3 commuting measurement
    # groups, so actual circuit executions ≈ samples × reservoirs × 3.
    n_meas_groups = 3  # approximate: {Z+ZZ}, {X}, {Y}
    n_individual_circuits = n_samples * n_reservoirs * n_meas_groups
    total_shots = n_individual_circuits * shots

    print(f"\n{'=' * 60}")
    print("COST ESTIMATION (before execution)")
    print(f"{'=' * 60}")
    print(f"  Backend:            {backend}")
    print(f"  Qubits:             {N_QUBITS}")
    print(f"  Samples:            {n_samples} (train + test)")
    print(f"  Reservoirs:         {n_reservoirs}")
    print(f"  Meas. groups:       ~{n_meas_groups} (non-commuting observable splits)")
    print(f"  Total circuits:     ~{n_individual_circuits:,}")
    print(f"  Shots per circuit:  {shots:,}")
    print(f"  Total shots:        ~{total_shots:,}")

    if backend == "sv1":
        # SV1 pricing: $0.075 per minute of simulation time
        est_sec_per_task = 0.005  # ~5 ms for a 4-qubit circuit
        est_minutes = (n_individual_circuits * est_sec_per_task) / 60
        est_cost = est_minutes * 0.075
        print(f"  Est. sim time:      ~{est_minutes:.2f} min")
        print(f"  Est. cost:          ~${est_cost:.4f} USD")
    elif backend in ("iqm", "rigetti"):
        pricing = {
            "iqm":     (0.30, 0.00145),
            "rigetti": (0.30, 0.0009),
        }
        per_task, per_shot = pricing[backend]
        task_cost = n_individual_circuits * per_task
        shot_cost = total_shots * per_shot
        est_cost = task_cost + shot_cost
        print(f"  Per-task price:     ${per_task}/task")
        print(f"  Per-shot price:     ${per_shot}/shot")
        print(f"  Task cost:          ~${task_cost:,.2f} USD")
        print(f"  Shot cost:          ~${shot_cost:,.2f} USD")
        print(f"  Est. total cost:    ~${est_cost:,.2f} USD")
    elif backend == "local":
        print("  Est. cost:          $0.00 (local simulator)")
    else:
        print("  Est. cost:          See AWS Braket pricing for this QPU")

    print(f"{'=' * 60}\n")


# ── 5. Main ──────────────────────────���──────────────────────────────────
def main():
    if BACKEND not in DEVICE_MAP:
        print(f"Unknown backend: {BACKEND!r}. Choose from: {list(DEVICE_MAP.keys())}")
        sys.exit(1)

    # Create PennyLane device backed by Braket.
    # parallel=True lets the plugin submit batched tapes concurrently.
    pl_name, device_arn = DEVICE_MAP[BACKEND]
    if device_arn is not None:
        dev = qml.device(pl_name, device_arn=device_arn,
                         wires=N_QUBITS, shots=SHOTS, parallel=True)
    else:
        dev = qml.device(pl_name, wires=N_QUBITS, shots=SHOTS)

    # Generate small dataset (cache filename includes sizes, won't collide
    # with the main 10k/10k dataset).
    dgp = DGPConfig(n_train=N_TRAIN, n_test=N_TEST, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]

    # Scale and clip (same as main experiment pipeline)
    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)

    total_samples = len(X_train_s) + len(X_test_s)
    estimate_cost(total_samples, N_RESERVOIRS, SHOTS, BACKEND)

    resp = input("Proceed with execution? [y/N] ")
    if resp.strip().lower() != "y":
        print("Aborted.")
        return

    # Run with both cost tracker and device tracker (for task IDs + progress)
    cost_tracker = Tracker().start()

    # Progress callback: logs execution count in real time.
    # With shot-based devices, non-commuting observables (X, Y, Z) are split
    # into ~3 commuting measurement groups, so actual circuit executions are
    # roughly 3× the number of (samples × reservoirs).
    _n_meas_groups = 3  # approximate: {Z+ZZ}, {X}, {Y}
    _expected_circuits = total_samples * N_RESERVOIRS * _n_meas_groups
    def _progress_callback(totals, history, latest):
        n = totals.get("executions", 0)
        pct = min(100, n * 100 // _expected_circuits) if _expected_circuits else 0
        print(f"\r  Circuit {n}/~{_expected_circuits}  ({pct}%)", end="", flush=True)

    t0 = time.perf_counter()

    augmenter = ReservoirAugmenter(
        n_qubits=N_QUBITS,
        n_reservoirs=N_RESERVOIRS,
        n_layers=N_LAYERS,
        observables=OBSERVABLES,
        device=dev,
    )
    augmenter.fit(X_train_s, y_train)

    print("Running circuits on Braket...")
    with qml.Tracker(dev, callback=_progress_callback) as dev_tracker:
        train_result = augmenter.transform(X_train_s)
        test_result = augmenter.transform(X_test_s)
    print()  # newline after carriage-return progress

    elapsed = time.perf_counter() - t0
    cost_tracker.stop()

    # Collect all task IDs for lookup on the AWS console
    task_ids = dev_tracker.history.get("braket_task_id", [])
    if task_ids:
        print(f"  Braket task IDs:  {len(task_ids)} tasks submitted")
        print(f"  First task ID:   {task_ids[0]}")
        print(f"  Last task ID:    {task_ids[-1]}")

    # ── Save augmented features (avoid costly re-runs) ─────────────────
    # Filename encodes: backend, config, data size, shots, method
    feat_dir = FEATURES_DIR / BACKEND / f"seed_{SEED}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / f"{augmenter.name}_n{N_TRAIN}+{N_TEST}_s{SHOTS}_pennylane.npz"
    np.savez_compressed(
        feat_path,
        train=train_result.features,
        test=test_result.features,
    )

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"  Augmenter:       {augmenter.name}")
    print(f"  Train features:  {train_result.features.shape}")
    print(f"  Test features:   {test_result.features.shape}")
    print(f"  Wall clock:      {elapsed:.1f} s")
    print(f"  Circuit depth:   {train_result.circuit_depth}")
    print(f"  Gate count:      {train_result.gate_count}")
    print(f"  Saved features:  {feat_path}")

    # Ridge regression
    model = RidgeModel()
    pred = model.fit_predict(train_result.features, y_train, test_result.features)
    mse = float(np.mean((pred.y_pred - y_test) ** 2))
    print(f"  Ridge test MSE:  {mse:.4f}")

    # ── Actual cost ──────────────────────────────────────────────────────
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
