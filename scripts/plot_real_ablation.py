"""Grouped bar plots for the 2×2 correlation ablation study."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = Path("results/real")
FIGURES_DIR = Path("plots/real")

CELLS = [
    ("noregime_nocorr", "Baseline"),
    ("noregime_corr", "+ Corr Quantum"),
    ("regime_nocorr", "+ Regime"),
    ("regime_corr", "+ Both"),
]


def load_ablation(prefix: str = "ablation_full") -> pd.DataFrame:
    """Load all 4 cells × 5 seeds and return a single DataFrame."""
    dfs = []
    for cell_tag, cell_label in CELLS:
        for seed in SEEDS:
            path = RESULTS_DIR / f"{prefix}_{cell_tag}_s{seed}" / "summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["seed"] = seed
            df["cell"] = cell_label
            df["cell_tag"] = cell_tag
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No ablation results found for {prefix}_*")
    return pd.concat(dfs, ignore_index=True)


def aggregate_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (augmenter, cell) across seeds."""
    agg = raw.groupby(["augmenter_name", "cell"]).agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ic_mean=("pearson_r", "mean"),
        ic_std=("pearson_r", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg


# Augmenter display order and colors
AUGMENTER_ORDER = [
    "identity",
    "rff_10", "rff_30", "rff_50", "rff_96",
    "poly_deg2", "poly_deg2_interact", "interaction_log",
    "qunified_z_8q_3L_3ens", "qunified_z_8q_3L_3ens_pca",
]

AUGMENTER_COLORS = {
    "identity": "#9D755D",
    "rff_10": "#F58518", "rff_30": "#F58518", "rff_50": "#F58518", "rff_96": "#F58518",
    "poly_deg2": "#B279A2", "poly_deg2_interact": "#B279A2",
    "interaction_log": "#EECA3B",
    "qunified_z_8q_3L_3ens": "#4C78A8",
    "qunified_z_8q_3L_3ens_pca": "#72B7B2",
}

AUGMENTER_SHORT = {
    "identity": "identity",
    "rff_10": "rff_10", "rff_30": "rff_30", "rff_50": "rff_50", "rff_96": "rff_96",
    "poly_deg2": "poly_deg2", "poly_deg2_interact": "poly_interact",
    "interaction_log": "interact_log",
    "qunified_z_8q_3L_3ens": "q:z_8q (mod)",
    "qunified_z_8q_3L_3ens_pca": "q:z_8q (pca)",
}

# Cell bar colors (one bar group per cell)
CELL_COLORS = {
    "Baseline": "#636EFA",
    "+ Corr Quantum": "#EF553B",
    "+ Regime": "#00CC96",
    "+ Both": "#AB63FA",
}


def plot_grouped_bars(agg: pd.DataFrame, prefix: str = "ablation_full"):
    """Two grouped bar charts: MSE and IC, grouped by augmenter, colored by cell."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["OOS MSE by Method and Ablation Cell", "OOS IC by Method and Ablation Cell"],
        vertical_spacing=0.18,
    )

    cell_labels = [label for _, label in CELLS]

    for cell_label in cell_labels:
        cell_data = agg[agg["cell"] == cell_label]
        # Order augmenters
        cell_data = cell_data.set_index("augmenter_name").reindex(AUGMENTER_ORDER).reset_index()
        cell_data = cell_data.dropna(subset=["mse_mean"])

        x_labels = [AUGMENTER_SHORT.get(n, n) for n in cell_data["augmenter_name"]]

        # MSE bars
        fig.add_trace(go.Bar(
            x=x_labels,
            y=cell_data["mse_mean"],
            error_y=dict(type="data", array=cell_data["mse_std"], visible=True, thickness=1.5),
            name=cell_label,
            marker_color=CELL_COLORS[cell_label],
            legendgroup=cell_label,
            showlegend=True,
        ), row=1, col=1)

        # IC bars
        fig.add_trace(go.Bar(
            x=x_labels,
            y=cell_data["ic_mean"],
            error_y=dict(type="data", array=cell_data["ic_std"], visible=True, thickness=1.5),
            name=cell_label,
            marker_color=CELL_COLORS[cell_label],
            legendgroup=cell_label,
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        barmode="group",
        title=f"Correlation Ablation: Method Performance Across 2×2 Cells (5 seeds, Ridge)",
        height=900, width=1200,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )

    fig.update_xaxes(tickangle=-35, row=1, col=1)
    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_yaxes(title_text="MSE (mean ± std)", row=1, col=1)
    fig.update_yaxes(title_text="IC (mean ± std)", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    path = FIGURES_DIR / "ablation_grouped_bars.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")

    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path.with_suffix('.png')}")
    except Exception:
        pass


def plot_delta_from_baseline(agg: pd.DataFrame):
    """Bar chart showing the change in MSE and IC relative to the baseline cell."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pivot to get baseline values
    pivot_mse = agg.pivot(index="augmenter_name", columns="cell", values="mse_mean")
    pivot_ic = agg.pivot(index="augmenter_name", columns="cell", values="ic_mean")

    if "Baseline" not in pivot_mse.columns:
        print("  Skipping delta plot — no Baseline cell found")
        return

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["ΔMSE vs Baseline (negative = improved)", "ΔIC vs Baseline (positive = improved)"],
        vertical_spacing=0.18,
    )

    for cell_label in ["+ Corr Quantum", "+ Regime", "+ Both"]:
        if cell_label not in pivot_mse.columns:
            continue

        delta_mse = pivot_mse[cell_label] - pivot_mse["Baseline"]
        delta_ic = pivot_ic[cell_label] - pivot_ic["Baseline"]

        # Reorder
        delta_mse = delta_mse.reindex(AUGMENTER_ORDER).dropna()
        delta_ic = delta_ic.reindex(AUGMENTER_ORDER).dropna()

        x_labels = [AUGMENTER_SHORT.get(n, n) for n in delta_mse.index]

        fig.add_trace(go.Bar(
            x=x_labels, y=delta_mse.values,
            name=cell_label, marker_color=CELL_COLORS[cell_label],
            legendgroup=cell_label, showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=x_labels, y=delta_ic.values,
            name=cell_label, marker_color=CELL_COLORS[cell_label],
            legendgroup=cell_label, showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        barmode="group",
        title="Effect of Regime / Corr Quantum Features (Δ from Baseline)",
        height=900, width=1200,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
    )
    fig.update_xaxes(tickangle=-35, row=1, col=1)
    fig.update_xaxes(tickangle=-35, row=2, col=1)
    fig.update_yaxes(title_text="ΔMSE", row=1, col=1)
    fig.update_yaxes(title_text="ΔIC", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    path = FIGURES_DIR / "ablation_delta.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")

    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path.with_suffix('.png')}")
    except Exception:
        pass


def main(prefix: str = "ablation_full"):
    print(f"Loading {prefix}_* results...")
    raw = load_ablation(prefix)
    agg = aggregate_ablation(raw)

    print(f"\nAggregated: {agg['augmenter_name'].nunique()} augmenters × "
          f"{len(CELLS)} cells × {agg['n_seeds'].iloc[0]} seeds")

    # Print summary table
    pivot = agg.pivot(index="augmenter_name", columns="cell", values="mse_mean")
    pivot = pivot.reindex(AUGMENTER_ORDER).reindex(columns=[c[1] for c in CELLS])
    print(f"\nMSE by method and cell:")
    print(pivot.to_string(float_format="%.6f"))

    print(f"\nGenerating plots...")
    plot_grouped_bars(agg, prefix)
    plot_delta_from_baseline(agg)
    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ablation_full"
    main(prefix)
