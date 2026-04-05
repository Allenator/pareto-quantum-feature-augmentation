"""Compare quantum reservoir features across backends (exact sim vs hardware).

Loads saved .npz feature files, computes feature-level statistics, trains
Ridge regression on each, and prints a side-by-side comparison table.
Includes a raw-features-only baseline to show augmentation value.

Usage:
    uv run python scripts/compare_features.py <file1.npz> <file2.npz> [file3.npz ...]
    uv run python scripts/compare_features.py features/synthetic_hw/lightning/seed_42/reservoir_3x3_Z_n100+50_exact.npz \
                                               features/synthetic_hw/rigetti/seed_42/reservoir_3x3_Z_n100+50_s1000_packed.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

N_ORIG = 4  # number of original feature columns
CLIP_RANGE = 5.0
RESULTS_DIR = Path("results/synthetic_hw")


def load_features(path: str) -> dict:
    """Load .npz and return dict with train/test arrays and metadata."""
    data = np.load(path)
    name = Path(path).stem
    return {"name": name, "path": path, "train": data["train"], "test": data["test"]}


def make_raw_baseline(n_train: int, n_test: int, seed: int = 42) -> dict:
    """Build a raw-features-only dataset (no augmentation) for comparison."""
    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=seed)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test = data[0], data[1]
    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)
    return {"name": "raw_features_only", "path": None, "train": X_train_s, "test": X_test_s}


def fit_and_predict(features: dict, y_train, y_test):
    """Fit Ridge and return (mse, y_pred)."""
    model = RidgeModel()
    pred = model.fit_predict(features["train"], y_train, features["test"])
    mse = float(np.mean((pred.y_pred - y_test) ** 2))
    return mse, pred.y_pred


def compare_pair(a: dict, b: dict, y_train: np.ndarray, y_test: np.ndarray):
    """Print detailed feature-level comparison between two feature sets."""
    print(f"\n{'─' * 70}")
    print(f"  A: {a['name']}")
    print(f"  B: {b['name']}")
    print(f"{'─' * 70}")

    print(f"\n  Shapes:")
    print(f"    A train={a['train'].shape}  test={a['test'].shape}")
    print(f"    B train={b['train'].shape}  test={b['test'].shape}")

    # Original features check (only if both have augmented cols)
    n_orig = min(N_ORIG, a["train"].shape[1], b["train"].shape[1])
    orig_diff = np.abs(a["train"][:, :n_orig] - b["train"][:, :n_orig]).max()
    print(f"\n  Original features (first {n_orig} cols) max diff: {orig_diff:.2e}"
          f"  {'OK' if orig_diff < 1e-10 else 'MISMATCH'}")

    # Augmented feature comparison (only if both have augmented cols)
    a_has_aug = a["train"].shape[1] > N_ORIG
    b_has_aug = b["train"].shape[1] > N_ORIG
    if a_has_aug and b_has_aug:
        a_aug = a["train"][:, N_ORIG:]
        b_aug = b["train"][:, N_ORIG:]
        a_aug_test = a["test"][:, N_ORIG:]
        b_aug_test = b["test"][:, N_ORIG:]

        if a_aug.shape != b_aug.shape:
            print(f"\n  WARNING: augmented shapes differ — skipping feature comparison")
        else:
            diff_train = np.abs(a_aug - b_aug)
            diff_test = np.abs(a_aug_test - b_aug_test)

            print(f"\n  Augmented feature difference:")
            print(f"    {'':20s} {'mean|diff|':>12s} {'max|diff|':>12s} {'std|diff|':>12s}")
            print(f"    {'Train':20s} {diff_train.mean():12.4f} {diff_train.max():12.4f} {diff_train.std():12.4f}")
            print(f"    {'Test':20s} {diff_test.mean():12.4f} {diff_test.max():12.4f} {diff_test.std():12.4f}")

            n_feat = a_aug.shape[1]
            corrs = []
            for i in range(n_feat):
                if a_aug[:, i].std() > 1e-10 and b_aug[:, i].std() > 1e-10:
                    corrs.append(np.corrcoef(a_aug[:, i], b_aug[:, i])[0, 1])
                else:
                    corrs.append(float("nan"))
            corrs = np.array(corrs)
            valid = corrs[~np.isnan(corrs)]

            print(f"\n  Per-feature correlation (train, {n_feat} features):")
            if len(valid) > 0:
                print(f"    Mean:  {np.nanmean(corrs):.4f}")
                print(f"    Min:   {np.nanmin(corrs):.4f}")
                print(f"    Max:   {np.nanmax(corrs):.4f}")
                print(f"    Valid: {len(valid)}/{n_feat}")

            print(f"\n  Per-feature detail:")
            print(f"    {'Feature':>8s} {'mean|diff|':>12s} {'corr':>8s}")
            for i in range(n_feat):
                c = f"{corrs[i]:.4f}" if not np.isnan(corrs[i]) else "n/a"
                print(f"    {i:>8d} {diff_train[:, i].mean():12.4f} {c:>8s}")

    # Ridge regression
    mse_a, pred_a = fit_and_predict(a, y_train, y_test)
    mse_b, pred_b = fit_and_predict(b, y_train, y_test)

    print(f"\n  Ridge regression:")
    print(f"    A test MSE: {mse_a:.4f}")
    print(f"    B test MSE: {mse_b:.4f}")
    print(f"    Δ MSE:      {mse_b - mse_a:+.4f} ({'B worse' if mse_b > mse_a else 'B better'})")

    pred_corr = np.corrcoef(pred_a, pred_b)[0, 1]
    print(f"    Prediction correlation: {pred_corr:.4f}")


def prediction_comparison(datasets: list[dict], y_train, y_test):
    """Cross-compare predictions across all datasets including raw baseline."""
    print(f"\n{'─' * 70}")
    print("  PREDICTION ANALYSIS")
    print(f"{'─' * 70}")

    # Fit all
    results = []
    for d in datasets:
        mse, pred = fit_and_predict(d, y_train, y_test)
        results.append({"name": d["name"], "mse": mse, "pred": pred})

    # MSE table
    print(f"\n  Ridge test MSE:")
    for r in results:
        n_feat = next(d["train"].shape[1] for d in datasets if d["name"] == r["name"])
        print(f"    {r['mse']:.4f}  ({n_feat} feat)  {r['name']}")

    # Pairwise prediction correlations
    print(f"\n  Prediction correlations:")
    print(f"    {'':30s}", end="")
    for r in results:
        print(f" {r['name'][:12]:>12s}", end="")
    print()
    for i, ri in enumerate(results):
        print(f"    {ri['name'][:30]:30s}", end="")
        for j, rj in enumerate(results):
            if j <= i:
                print(f" {'':>12s}", end="")
            else:
                corr = np.corrcoef(ri["pred"], rj["pred"])[0, 1]
                print(f" {corr:12.4f}", end="")
        print()

    # Per-sample improvement over raw baseline
    raw_idx = next(i for i, r in enumerate(results) if r["name"] == "raw_features_only")
    err_raw = np.abs(results[raw_idx]["pred"] - y_test)

    print(f"\n  Per-sample |error| vs raw baseline:")
    for i, r in enumerate(results):
        if i == raw_idx:
            continue
        err = np.abs(r["pred"] - y_test)
        n_better = int((err < err_raw).sum())
        n_total = len(y_test)
        print(f"    {r['name'][:40]:40s}  better on {n_better}/{n_total} samples")

    # Residual analysis
    print(f"\n  Residual analysis (pred - y_test):")
    for r in results:
        resid = r["pred"] - y_test
        print(f"    {r['name'][:30]:30s}  mean={resid.mean():+.4f}  std={resid.std():.4f}"
              f"  min={resid.min():+.4f}  max={resid.max():+.4f}")


def export_csv(all_datasets: list[dict], y_train, y_test):
    """Export comparison results to CSV files for presentation.

    Produces three CSVs:
      - summary.csv: one row per dataset with MSE, feature count, residual stats
      - prediction_correlations.csv: pairwise prediction correlation matrix
      - feature_detail.csv: per-feature difference and correlation (pairwise, augmented files only)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = RidgeModel()

    # ── 1. summary.csv ───────────────────────────────��───────────────────
    results = []
    for d in all_datasets:
        mse, pred = fit_and_predict(d, y_train, y_test)
        resid = pred - y_test
        raw_mse = None
        if d["name"] != "raw_features_only":
            raw_d = next(x for x in all_datasets if x["name"] == "raw_features_only")
            raw_mse_val, _ = fit_and_predict(raw_d, y_train, y_test)
            raw_mse = raw_mse_val
        results.append({
            "dataset": d["name"],
            "n_features": d["train"].shape[1],
            "n_augmented": d["train"].shape[1] - N_ORIG,
            "n_train": d["train"].shape[0],
            "n_test": d["test"].shape[0],
            "ridge_mse": round(mse, 4),
            "mse_vs_raw": round(mse - raw_mse, 4) if raw_mse is not None else "",
            "mse_pct_change_vs_raw": round((mse - raw_mse) / raw_mse * 100, 2) if raw_mse is not None else "",
            "residual_mean": round(float(resid.mean()), 4),
            "residual_std": round(float(resid.std()), 4),
            "residual_min": round(float(resid.min()), 4),
            "residual_max": round(float(resid.max()), 4),
        })

    summary_path = RESULTS_DIR / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── 2. prediction_correlations.csv ────────────────────────────────────
    preds = []
    names = []
    for d in all_datasets:
        _, pred = fit_and_predict(d, y_train, y_test)
        preds.append(pred)
        names.append(d["name"])

    corr_path = RESULTS_DIR / "prediction_correlations.csv"
    with open(corr_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for i, name_i in enumerate(names):
            row = [name_i]
            for j in range(len(names)):
                row.append(round(float(np.corrcoef(preds[i], preds[j])[0, 1]), 4))
            writer.writerow(row)

    # ── 3. feature_detail.csv ───────────────────────────────���─────────────
    # Pairwise: first augmented file vs all other augmented files
    augmented = [d for d in all_datasets if d["train"].shape[1] > N_ORIG]
    detail_rows = []
    if len(augmented) >= 2:
        ref = augmented[0]
        for other in augmented[1:]:
            ref_aug = ref["train"][:, N_ORIG:]
            other_aug = other["train"][:, N_ORIG:]
            if ref_aug.shape != other_aug.shape:
                continue
            for i in range(ref_aug.shape[1]):
                diff = np.abs(ref_aug[:, i] - other_aug[:, i])
                corr = float("nan")
                if ref_aug[:, i].std() > 1e-10 and other_aug[:, i].std() > 1e-10:
                    corr = float(np.corrcoef(ref_aug[:, i], other_aug[:, i])[0, 1])
                detail_rows.append({
                    "reference": ref["name"],
                    "compared_to": other["name"],
                    "feature_idx": i,
                    "mean_abs_diff": round(float(diff.mean()), 4),
                    "max_abs_diff": round(float(diff.max()), 4),
                    "std_abs_diff": round(float(diff.std()), 4),
                    "correlation": round(corr, 4) if not np.isnan(corr) else "",
                })

    detail_path = RESULTS_DIR / "feature_detail.csv"
    if detail_rows:
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)

    print(f"\n  CSV files exported to {RESULTS_DIR}/:")
    print(f"    {summary_path}")
    print(f"    {corr_path}")
    if detail_rows:
        print(f"    {detail_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: compare_features.py <file1.npz> [file2.npz ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    datasets = [load_features(p) for p in paths]

    # Infer data sizes from the first file
    n_train = datasets[0]["train"].shape[0]
    n_test = datasets[0]["test"].shape[0]

    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=42)
    data = get_or_generate(dgp, "data/synthetic")
    y_train, y_test = data[2], data[3]

    # Add raw baseline
    raw = make_raw_baseline(n_train, n_test)
    all_datasets = [raw] + datasets

    # Header
    print(f"\n{'=' * 70}")
    print("FEATURE COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Data: {n_train} train / {n_test} test")
    print(f"  Datasets:")
    for i, d in enumerate(all_datasets):
        print(f"    [{i}] {d['name']:45s}  shape={d['train'].shape}")

    # Prediction comparison across all (including raw)
    prediction_comparison(all_datasets, y_train, y_test)

    # Pairwise feature comparisons (first loaded file vs all others)
    if len(datasets) >= 2:
        ref = datasets[0]
        for other in datasets[1:]:
            compare_pair(ref, other, y_train, y_test)

    # Export CSVs
    export_csv(all_datasets, y_train, y_test)

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
