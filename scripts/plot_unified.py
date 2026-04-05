"""Plot unified factorial results: MSE vs # features, colored by design dimensions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = Path("results/synthetic/unified_factorial")
FIGURES_DIR = Path("plots/synthetic/unified_factorial/figures")


def load_results() -> pd.DataFrame:
    records = []
    for p in RESULTS_DIR.rglob("*.json"):
        with open(p) as f:
            records.append(json.load(f))
    if not records:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")
    return pd.DataFrame(records)


def parse_config(name: str) -> dict:
    d = {"encoding": "", "connectivity": "", "cnot_mixing": False,
         "observables": "Z", "random_rot": False, "n_layers": 1, "n_ensemble": 1}
    parts = name.replace("unified_", "")
    for enc in ["RZ", "IQP", "angle"]:
        if parts.startswith(enc + "_"):
            d["encoding"] = enc
            parts = parts[len(enc) + 1:]
            break
    for conn in ["linear", "circular", "all"]:
        if parts.startswith(conn + "_"):
            d["connectivity"] = conn
            parts = parts[len(conn) + 1:]
            break
    if parts.startswith("cnot"):
        d["cnot_mixing"] = True
        parts = parts[4:]
    elif parts.startswith("noc"):
        d["cnot_mixing"] = False
        parts = parts[3:]
    for obs in ["_Z+ZZ", "_XYZ", "_full", "_prob"]:
        if parts.startswith(obs):
            d["observables"] = obs[1:]
            parts = parts[len(obs):]
            break
    if "_rot" in parts:
        d["random_rot"] = True
        parts = parts.replace("_rot", "", 1)
    m = re.search(r"_(\d+)L", parts)
    if m:
        d["n_layers"] = int(m.group(1))
    m = re.search(r"_(\d+)ens", parts)
    if m:
        d["n_ensemble"] = int(m.group(1))
    return d


def build_df(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["augmenter_name"].apply(parse_config).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df["cnot"] = df["cnot_mixing"].map({True: "CNOT", False: "no CNOT"})
    df["rot"] = df["random_rot"].map({True: "random Rot", False: "no Rot"})
    df["structure"] = df["cnot"] + " + " + df["rot"]
    df["hover"] = (
        "<b>" + df["augmenter_name"] + "</b><br>"
        + "Encoding: " + df["encoding"] + "<br>"
        + "Connectivity: " + df["connectivity"] + "<br>"
        + "CNOT: " + df["cnot"] + "<br>"
        + "Observables: " + df["observables"] + "<br>"
        + "Random Rot: " + df["rot"] + "<br>"
        + "Layers: " + df["n_layers"].astype(str) + "<br>"
        + "Ensemble: " + df["n_ensemble"].astype(str) + "<br>"
        + "Features: " + df["n_features"].astype(str) + "<br>"
        + "MSE: " + df["mse_mean"].round(3).astype(str)
        + " ± " + df["mse_std"].fillna(0).round(3).astype(str) + "<br>"
        + "MSE R1: " + df["mse_r1_mean"].round(3).astype(str) + "<br>"
        + "MSE R2: " + df["mse_r2_mean"].round(3).astype(str) + "<br>"
        + "Random params: " + df["n_random_params"].fillna(0).astype(int).astype(str)
    )
    return df


def _pareto_frontier(df_sub):
    """Pareto frontier: sweep left-to-right on n_features, track best MSE."""
    df_sorted = df_sub.sort_values("n_features").reset_index(drop=True)
    frontier = []
    best_mse = float("inf")
    for _, row in df_sorted.iterrows():
        if row["mse_mean"] < best_mse:
            best_mse = row["mse_mean"]
            frontier.append(row)
    return pd.DataFrame(frontier)


# Tableau 10 — muted tones that don't clash with red/green reference lines
_COLORS = [
    "#4C78A8",  # steel blue
    "#F58518",  # orange
    "#72B7B2",  # teal
    "#B279A2",  # mauve
    "#EECA3B",  # gold
    "#9D755D",  # brown
    "#FF9DA6",  # pink
    "#BAB0AC",  # warm gray
    "#54A24B",  # sage
    "#E45756",  # coral
]

# Reference baselines (computed on 2K/2K data, Ridge CV, seed 42)
_IDENTITY_MSE = 4.7693
_ORACLE_MSE = 2.4149


def _add_reference_lines(fig, row=None, col=None):
    """Add identity/oracle/noise floor lines with annotations below each."""
    kwargs = {}
    if row is not None and col is not None:
        kwargs = {"row": row, "col": col}

    lines = [
        (_IDENTITY_MSE, "Identity baseline", "#d62728"),
        (_ORACLE_MSE, "Oracle (X₁X₃ + log|X₂|)", "#2ca02c"),
        (1.0, "Noise floor (Var(ε) = 1)", "#888888"),
    ]
    for y_val, label, color in lines:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1.5,
                      opacity=0.6, **kwargs)

    # Annotations — use xref based on whether we're in a subplot
    if row is not None and col is not None:
        # Subplot: use axis-domain x, axis y
        ax_suffix = "" if (row == 1 and col == 1) else str((row - 1) * 2 + col)
        xref = f"x{ax_suffix} domain" if ax_suffix else "x domain"
        yref = f"y{ax_suffix}" if ax_suffix else "y"
        for y_val, label, color in lines:
            fig.add_annotation(
                x=0.98, y=y_val, xref=xref, yref=yref,
                text=label, showarrow=False, xanchor="right", yanchor="top",
                yshift=-3, font=dict(size=8, color=color),
            )
    else:
        for y_val, label, color in lines:
            fig.add_annotation(
                x=0.98, y=y_val, xref="paper", yref="y",
                text=label, showarrow=False, xanchor="right", yanchor="top",
                yshift=-3, font=dict(size=9, color=color),
            )


def make_scatter(df, color_col, title, filename):
    """MSE vs features scatter (mean across seeds) with Pareto frontier + error bars."""
    categories = df[color_col].unique()
    fig = go.Figure()

    for i, cat in enumerate(sorted(categories, key=str)):
        sub = df[df[color_col] == cat]
        color = _COLORS[i % len(_COLORS)]

        # Scatter points (mean across seeds)
        fig.add_trace(go.Scatter(
            x=sub["n_features"], y=sub["mse_mean"],
            mode="markers", name=str(cat),
            legendgroup=str(cat),
            marker=dict(size=5, color=color, opacity=0.3),
            hovertext=sub["hover"], hoverinfo="text",
        ))

        # Pareto frontier with std error bars
        frontier = _pareto_frontier(sub)
        if len(frontier) > 1:
            error_y = None
            if "mse_std" in frontier.columns and frontier["mse_std"].notna().any():
                error_y = dict(type="data", array=frontier["mse_std"].fillna(0),
                               visible=True, thickness=1.5, width=4)
            fig.add_trace(go.Scatter(
                x=frontier["n_features"], y=frontier["mse_mean"],
                error_y=error_y,
                mode="lines+markers", name=f"Pareto: {cat}",
                legendgroup=str(cat), showlegend=False,
                marker=dict(size=8, color=color, symbol="diamond"),
                line=dict(color=color, width=2),
                hovertext=frontier["hover"], hoverinfo="text",
            ))

    _add_reference_lines(fig)
    fig.update_layout(
        title=title,
        xaxis=dict(title="# Features", type="log"),
        yaxis=dict(title="Test MSE"),
        height=600, width=1000,
        template="plotly_white",
        hovermode="closest",
    )
    path = FIGURES_DIR / filename
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def make_2x2_structure(df):
    """2x2 grid: encoding colored, faceted by structure (CNOT × Rot)."""
    from plotly.subplots import make_subplots

    # 2x2: rows = CNOT (no/yes), cols = Rot (no/yes)
    structures = [
        (1, 1, "no CNOT + no Rot"),
        (1, 2, "no CNOT + random Rot"),
        (2, 1, "CNOT + no Rot"),
        (2, 2, "CNOT + random Rot"),
    ]
    categories = sorted(df["encoding"].unique(), key=str)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s[2] for s in structures],
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    legend_added = set()
    for row, col, struct_name in structures:
        facet_df = df[df["structure"] == struct_name]
        for i, cat in enumerate(categories):
            sub = facet_df[facet_df["encoding"] == cat]
            if sub.empty:
                continue
            color = _COLORS[i % len(_COLORS)]
            show_legend = cat not in legend_added
            legend_added.add(cat)

            fig.add_trace(go.Scatter(
                x=sub["n_features"], y=sub["mse_mean"],
                mode="markers", name=str(cat),
                legendgroup=str(cat), showlegend=show_legend,
                marker=dict(size=4, color=color, opacity=0.3),
                hovertext=sub["hover"], hoverinfo="text",
            ), row=row, col=col)

            frontier = _pareto_frontier(sub)
            if len(frontier) > 1:
                error_y = None
                if "mse_std" in frontier.columns and frontier["mse_std"].notna().any():
                    error_y = dict(type="data", array=frontier["mse_std"].fillna(0),
                                   visible=True, thickness=1.5, width=3)
                fig.add_trace(go.Scatter(
                    x=frontier["n_features"], y=frontier["mse_mean"],
                    error_y=error_y,
                    mode="lines+markers",
                    legendgroup=str(cat), showlegend=False,
                    marker=dict(size=7, color=color, symbol="diamond"),
                    line=dict(color=color, width=2),
                    hovertext=frontier["hover"], hoverinfo="text",
                ), row=row, col=col)

        _add_reference_lines(fig, row=row, col=col)

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="# Features", type="log", row=row, col=col)
            fig.update_yaxes(title_text="Test MSE", matches="y", row=row, col=col)

    fig.update_layout(
        title="MSE vs Features — Encoding × Structure (2×2)",
        height=900, width=1100,
        template="plotly_white",
        hovermode="closest",
    )
    fig.for_each_annotation(lambda a: a.update(font_size=12))

    path = FIGURES_DIR / "encoding_x_structure.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        raw = load_results()
        df = raw.groupby("augmenter_name").agg(
            n_features=("n_features_total", "first"),
            mse_mean=("mse", "mean"),
            mse_r1_mean=("mse_regime1", "mean"),
            mse_r2_mean=("mse_regime2", "mean"),
            n_random_params=("n_random_params", "first"),
        ).reset_index()

    df = build_df(df)

    # Individual plots
    make_scatter(df, "encoding", "MSE vs Features — by Encoding", "by_encoding.html")
    make_scatter(df, "structure", "MSE vs Features — by Circuit Structure", "by_structure.html")
    make_2x2_structure(df)
    make_scatter(df, "observables", "MSE vs Features — by Observables", "by_observables.html")

    df["n_ensemble_str"] = df["n_ensemble"].astype(str) + " circuit(s)"
    make_scatter(df, "n_ensemble_str", "MSE vs Features — by Ensemble Size", "by_ensemble.html")

    df["n_layers_str"] = df["n_layers"].astype(str) + " layer(s)"
    make_scatter(df, "n_layers_str", "MSE vs Features — by # Layers", "by_layers.html")

    make_scatter(df, "connectivity", "MSE vs Features — by Connectivity", "by_connectivity.html")
    make_scatter(df, "rot", "MSE vs Features — by Randomization", "by_randomization.html")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
