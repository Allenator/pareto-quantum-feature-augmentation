"""Plot hardware (Rigetti Ankaa-3) vs exact simulator vs raw baseline.

Produces two Plotly figures (HTML + PNG) in the teammate's `plot_pareto_vs_classical`
style, but adapted for the small-data hardware experiment:

  1. MSE vs # Features (one point per method × dataset size)
     — shows where hardware lands on the features/MSE plane
  2. MSE vs Training Set Size
     — shows the scaling gap between ideal and real QPU

Data sources:
  features/synthetic_hw/lightning/seed_42/reservoir_3x3_Z_n{N}+{M}_exact.npz
  features/synthetic_hw/rigetti/seed_42/reservoir_3x3_Z_n{N}+{M}_s1000_packed.npz

Usage:
    uv run python scripts/plot_synthetic_hw.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

FEATURES_DIR = Path("features/synthetic_hw")
FIGURES_DIR = Path("plots/synthetic_hw")
CLIP_RANGE = 5.0
SEED = 42

# Dataset sizes (n_train, n_test) with features available on both backends
SIZES = [(100, 50), (500, 250), (1000, 500)]

# Matches the teammate's color palette in plot_pareto_vs_classical.py
COLORS = {
    "raw":     "#9D755D",   # brown — identity / raw baseline
    "exact":   "#4C78A8",   # steel blue — Quantum Pareto equivalent
    "rigetti": "#E45756",   # coral — hardware
}


def load_and_eval(path: Path, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Load features, fit Ridge, return MSE + metadata."""
    data = np.load(path)
    train, test = data["train"], data["test"]
    pred = RidgeModel().fit_predict(train, y_train, test)
    return {
        "n_features": train.shape[1],
        "n_augmented": train.shape[1] - 4,
        "n_test": test.shape[0],
        "ridge_mse": float(np.mean((pred.y_pred - y_test) ** 2)),
        "y_pred": pred.y_pred,
    }


def raw_baseline(n_train: int, n_test: int) -> dict:
    """Raw features only (no augmentation). Returns same schema as load_and_eval."""
    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(X_train), -CLIP_RANGE, CLIP_RANGE)
    X_test_s = np.clip(scaler.transform(X_test), -CLIP_RANGE, CLIP_RANGE)
    pred = RidgeModel().fit_predict(X_train_s, y_train, X_test_s)
    return {
        "n_features": X_train_s.shape[1],
        "n_augmented": 0,
        "n_test": n_test,
        "ridge_mse": float(np.mean((pred.y_pred - y_test) ** 2)),
        "y_pred": pred.y_pred,
    }


def gather_results() -> dict:
    """Compute MSE for raw/exact/rigetti at each (n_train, n_test) size."""
    out = {"raw": [], "exact": [], "rigetti": []}
    for n_train, n_test in SIZES:
        dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
        data = get_or_generate(dgp, "data/synthetic")
        y_train, y_test = data[2], data[3]

        # Raw baseline
        r = raw_baseline(n_train, n_test)
        r["n_train"] = n_train
        r["label"] = f"Raw ({n_train}/{n_test})"
        out["raw"].append(r)

        # Exact simulator (lightning)
        path = FEATURES_DIR / "lightning" / f"seed_{SEED}" / \
            f"reservoir_3x3_Z_n{n_train}+{n_test}_exact.npz"
        if path.exists():
            e = load_and_eval(path, y_train, y_test)
            e["n_train"] = n_train
            e["label"] = f"Exact ({n_train}/{n_test})"
            out["exact"].append(e)

        # Rigetti hardware
        path = FEATURES_DIR / "rigetti" / f"seed_{SEED}" / \
            f"reservoir_3x3_Z_n{n_train}+{n_test}_s1000_packed.npz"
        if path.exists():
            h = load_and_eval(path, y_train, y_test)
            h["n_train"] = n_train
            h["label"] = f"Rigetti ({n_train}/{n_test})"
            out["rigetti"].append(h)
    return out


def hover_text(row: dict) -> str:
    return (
        f"<b>{row['label']}</b><br>"
        f"n_train = {row['n_train']}<br>"
        f"Features: {row['n_features']}<br>"
        f"Ridge MSE: {row['ridge_mse']:.4f}"
    )


def _add_method_traces(fig, rows_by_method, x_key, size_to_symbol,
                       line_dash="solid"):
    """Add a line+markers trace for each method (Raw/Exact/Rigetti)."""
    for method, rows, color in rows_by_method:
        if not rows:
            continue
        xs = [r[x_key] for r in rows]
        ys = [r["ridge_mse"] for r in rows]
        hovers = [hover_text(r) for r in rows]
        symbols = [size_to_symbol[r["n_train"]] for r in rows]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+lines",
            name=method,
            marker=dict(size=13, color=color, symbol=symbols,
                        line=dict(color="white", width=1)),
            line=dict(color=color, width=2.5, dash=line_dash),
            hovertext=hovers, hoverinfo="text",
            showlegend=True,
            legendgroup=method,
        ))


def _save_fig(fig, path: Path):
    fig.write_html(str(path), include_plotlyjs=True)
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path} + .png")
    except Exception as e:
        print(f"Saved: {path}  (PNG export failed: {e})")


def make_plots(results: dict):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = results["raw"]
    exact = results["exact"]
    rigetti = results["rigetti"]
    rows_by_method = [
        ("Raw", raw, COLORS["raw"]),
        ("Exact simulator", exact, COLORS["exact"]),
        ("Rigetti Ankaa-3", rigetti, COLORS["rigetti"]),
    ]
    size_to_symbol = {100: "circle", 500: "square", 1000: "diamond"}

    common_layout = dict(
        height=520, width=900,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.2,
                    xanchor="center", x=0.5),
    )

    # ── Figure 1: MSE vs # Features ───────────────────────────────────────
    fig1 = go.Figure()
    _add_method_traces(fig1, rows_by_method, "n_features", size_to_symbol,
                       line_dash="dot")
    fig1.add_hline(y=1.0, line_dash="dash", line_color="#888888",
                   line_width=1.5, opacity=0.6)
    fig1.add_annotation(x=0.98, y=1.0, xref="x domain", yref="y",
                        text="Noise floor (σ²=1)", showarrow=False,
                        xanchor="right", yanchor="top",
                        yshift=-3, font=dict(size=10, color="#888888"))
    # Marker-size sub-legend
    for n_train, symbol in size_to_symbol.items():
        fig1.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=13, color="#666666", symbol=symbol,
                        line=dict(color="white", width=1)),
            name=f"n_train = {n_train}", legendgroup="sizes", showlegend=True,
        ))
    fig1.update_xaxes(title_text="# Features (log)", type="log")
    fig1.update_yaxes(title_text="Ridge Test MSE")
    fig1.update_layout(
        title="MSE vs # Features — Reservoir 3×3 Z Augmentation",
        **common_layout,
    )
    _save_fig(fig1, FIGURES_DIR / "mse_vs_features.html")

    # ── Figure 2: MSE vs Training Size ──────────────────────────��─────────
    fig2 = go.Figure()
    _add_method_traces(fig2, rows_by_method, "n_train", size_to_symbol,
                       line_dash="solid")
    fig2.add_hline(y=1.0, line_dash="dash", line_color="#888888",
                   line_width=1.5, opacity=0.6)
    fig2.add_annotation(x=0.98, y=1.0, xref="x domain", yref="y",
                        text="Noise floor (σ²=1)", showarrow=False,
                        xanchor="right", yanchor="top",
                        yshift=-3, font=dict(size=10, color="#888888"))
    fig2.update_xaxes(title_text="Training Set Size (log)", type="log")
    fig2.update_yaxes(title_text="Ridge Test MSE")
    fig2.update_layout(
        title="MSE vs Training Size — Reservoir 3×3 Z Augmentation",
        **common_layout,
    )
    _save_fig(fig2, FIGURES_DIR / "mse_vs_training_size.html")


def print_table(results: dict):
    """Text summary table of all results."""
    print(f"\n{'=' * 72}")
    print("HARDWARE vs SIMULATION — Ridge test MSE")
    print(f"{'=' * 72}")
    print(f"  {'Method':<20s} {'n_train':>8s} {'n_test':>8s} "
          f"{'n_feat':>8s} {'MSE':>10s}")
    for method in ("raw", "exact", "rigetti"):
        for r in results[method]:
            print(f"  {method:<20s} {r['n_train']:>8d} "
                  f"{r.get('n_test', '—'):>8} "
                  f"{r['n_features']:>8d} {r['ridge_mse']:>10.4f}")
    print(f"{'=' * 72}")


def main():
    results = gather_results()
    print_table(results)
    make_plots(results)


if __name__ == "__main__":
    main()
