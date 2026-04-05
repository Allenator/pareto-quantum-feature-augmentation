"""Plot unified factorial results: MSE vs # features, colored by design dimensions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import re

import numpy as np
import pandas as pd
import plotly.express as px

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
    """Extract design dimensions from unified augmenter name."""
    # unified_{enc}_{conn}_{cnot}[_{obs}][_rot][_{n}L][_{e}ens]
    d = {"encoding": "", "connectivity": "", "cnot_mixing": False,
         "observables": "Z", "random_rot": False, "n_layers": 1, "n_ensemble": 1}

    parts = name.replace("unified_", "")

    # Encoding
    for enc in ["RZ", "IQP", "angle"]:
        if parts.startswith(enc + "_"):
            d["encoding"] = enc
            parts = parts[len(enc) + 1:]
            break

    # Connectivity
    for conn in ["linear", "circular", "all"]:
        if parts.startswith(conn + "_"):
            d["connectivity"] = conn
            parts = parts[len(conn) + 1:]
            break

    # CNOT mixing
    if parts.startswith("cnot"):
        d["cnot_mixing"] = True
        parts = parts[4:]
    elif parts.startswith("noc"):
        d["cnot_mixing"] = False
        parts = parts[3:]

    # Observables
    for obs in ["_Z+ZZ", "_XYZ", "_full", "_prob"]:
        if parts.startswith(obs):
            d["observables"] = obs[1:]
            parts = parts[len(obs):]
            break

    # Random rot
    if "_rot" in parts:
        d["random_rot"] = True
        parts = parts.replace("_rot", "", 1)

    # Layers
    m = re.search(r"_(\d+)L", parts)
    if m:
        d["n_layers"] = int(m.group(1))

    # Ensemble
    m = re.search(r"_(\d+)ens", parts)
    if m:
        d["n_ensemble"] = int(m.group(1))

    return d


def build_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse configs and add design dimension columns."""
    parsed = df["augmenter_name"].apply(parse_config).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Readable labels
    df["cnot"] = df["cnot_mixing"].map({True: "CNOT", False: "no CNOT"})
    df["rot"] = df["random_rot"].map({True: "random Rot", False: "no Rot"})
    df["structure"] = df["cnot"] + " + " + df["rot"]

    # Hover
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
        + "MSE: " + df["mse_mean"].round(3).astype(str) + "<br>"
        + "MSE R1: " + df["mse_r1_mean"].round(3).astype(str) + "<br>"
        + "MSE R2: " + df["mse_r2_mean"].round(3).astype(str) + "<br>"
        + "Random params: " + df["n_random_params"].fillna(0).astype(int).astype(str)
    )
    return df


def make_scatter(df, color_col, title, filename):
    """MSE vs features scatter, colored by a design dimension."""
    fig = px.scatter(
        df, x="n_features", y="mse_mean",
        color=color_col,
        custom_data=["hover"],
        log_x=True,
        labels={"n_features": "# Features", "mse_mean": "Test MSE", color_col: color_col},
        title=title,
        opacity=0.6,
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(size=6),
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=600, width=1000,
        template="plotly_white",
        hovermode="closest",
    )
    path = FIGURES_DIR / filename
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def make_faceted_scatter(df, color_col, facet_col, title, filename):
    """MSE vs features, colored by one dim, faceted by another."""
    fig = px.scatter(
        df, x="n_features", y="mse_mean",
        color=color_col,
        facet_col=facet_col,
        custom_data=["hover"],
        log_x=True,
        labels={"n_features": "# Features", "mse_mean": "Test MSE"},
        title=title,
        opacity=0.6,
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(size=5),
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=600, width=1400,
        template="plotly_white",
        hovermode="closest",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    path = FIGURES_DIR / filename
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_results()
    # Use summary if available, otherwise aggregate
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        df = raw.groupby("augmenter_name").agg(
            n_features=("n_features_total", "first"),
            mse_mean=("mse", "mean"),
            mse_r1_mean=("mse_regime1", "mean"),
            mse_r2_mean=("mse_regime2", "mean"),
            n_random_params=("n_random_params", "first"),
        ).reset_index()

    df = build_df(df)

    # 1. Color by encoding
    make_scatter(df, "encoding", "MSE vs Features — by Encoding", "by_encoding.html")

    # 2. Color by structure (CNOT × Rot combination)
    make_scatter(df, "structure", "MSE vs Features — by Circuit Structure", "by_structure.html")

    # 3. Color by encoding, faceted by structure
    make_faceted_scatter(df, "encoding", "structure",
                         "MSE vs Features — Encoding × Structure", "encoding_x_structure.html")

    # 4. Color by observables
    make_scatter(df, "observables", "MSE vs Features — by Observables", "by_observables.html")

    # 5. Color by n_ensemble
    df["n_ensemble_str"] = df["n_ensemble"].astype(str) + " circuit(s)"
    make_scatter(df, "n_ensemble_str", "MSE vs Features — by Ensemble Size", "by_ensemble.html")

    # 6. Color by n_layers
    df["n_layers_str"] = df["n_layers"].astype(str) + " layer(s)"
    make_scatter(df, "n_layers_str", "MSE vs Features — by # Layers", "by_layers.html")

    # 7. Color by connectivity
    make_scatter(df, "connectivity", "MSE vs Features — by Connectivity", "by_connectivity.html")

    # 8. Pareto frontier: no randomization vs randomization
    make_pareto(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


def _pareto_frontier(df_sub):
    """Compute Pareto frontier: minimize both n_features and mse_mean."""
    df_sorted = df_sub.sort_values("n_features").reset_index(drop=True)
    frontier = []
    best_mse = float("inf")
    for _, row in df_sorted.iterrows():
        if row["mse_mean"] < best_mse:
            best_mse = row["mse_mean"]
            frontier.append(row)
    return pd.DataFrame(frontier)


def make_pareto(df):
    """Pareto frontiers for randomized vs non-randomized augmenters."""
    import plotly.graph_objects as go

    df_no_rot = df[~df["random_rot"]].copy()
    df_rot = df[df["random_rot"]].copy()

    frontier_no_rot = _pareto_frontier(df_no_rot)
    frontier_rot = _pareto_frontier(df_rot)

    fig = go.Figure()

    # Background scatter — no rot
    fig.add_trace(go.Scatter(
        x=df_no_rot["n_features"], y=df_no_rot["mse_mean"],
        mode="markers", name="No randomization",
        marker=dict(size=5, color="#1f77b4", opacity=0.2),
        hovertext=df_no_rot["hover"], hoverinfo="text",
    ))

    # Background scatter — rot
    fig.add_trace(go.Scatter(
        x=df_rot["n_features"], y=df_rot["mse_mean"],
        mode="markers", name="With randomization",
        marker=dict(size=5, color="#d62728", opacity=0.2),
        hovertext=df_rot["hover"], hoverinfo="text",
    ))

    # Pareto frontier — no rot
    fig.add_trace(go.Scatter(
        x=frontier_no_rot["n_features"], y=frontier_no_rot["mse_mean"],
        mode="lines+markers", name="Pareto — no randomization",
        marker=dict(size=10, color="#1f77b4", symbol="circle"),
        line=dict(color="#1f77b4", width=2),
        hovertext=frontier_no_rot["hover"], hoverinfo="text",
    ))

    # Pareto frontier — rot
    fig.add_trace(go.Scatter(
        x=frontier_rot["n_features"], y=frontier_rot["mse_mean"],
        mode="lines+markers", name="Pareto — with randomization",
        marker=dict(size=10, color="#d62728", symbol="diamond"),
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertext=frontier_rot["hover"], hoverinfo="text",
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Pareto Frontier: MSE vs Features — Randomized vs Non-randomized",
        xaxis=dict(title="# Features", type="log"),
        yaxis=dict(title="Test MSE"),
        height=650, width=1100,
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    path = FIGURES_DIR / "pareto_frontier.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
