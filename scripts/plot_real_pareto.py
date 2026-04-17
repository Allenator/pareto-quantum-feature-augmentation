"""Pareto plots for real data backtest results across 5 seeds."""

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


def load_multi_seed(run_prefix: str) -> pd.DataFrame:
    """Load and aggregate summary CSVs across seeds."""
    dfs = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{run_prefix}_s{seed}" / "summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No results found for {run_prefix}_s*")
    return pd.concat(dfs, ignore_index=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-augmenter metrics across seeds."""
    agg = raw.groupby("augmenter_name").agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        ic_mean=("pearson_r", "mean"),
        ic_std=("pearson_r", "std"),
        n_features=("n_features_total", "first"),
        n_random_params=("n_random_params", "first"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg


def classify(name: str) -> str:
    if name.startswith("qunified_"):
        return "Quantum"
    if name.startswith("rff_"):
        return "RFF"
    if name.startswith("poly_") or name == "poly_deg2":
        return "Polynomial"
    if name == "interaction_log":
        return "Interaction+Log"
    return "Identity"


def _pareto_frontier(df, metric_col, lower_is_better=True):
    """Extract Pareto-optimal points (feature count vs metric)."""
    df_sorted = df.sort_values("n_features").reset_index(drop=True)
    frontier = []
    best = float("inf") if lower_is_better else float("-inf")
    for _, row in df_sorted.iterrows():
        val = row[metric_col]
        if (lower_is_better and val < best) or (not lower_is_better and val > best):
            best = val
            frontier.append(row)
    return pd.DataFrame(frontier)


def hover_text(row):
    return (
        f"<b>{row['augmenter_name']}</b><br>"
        f"Features: {row['n_features']}<br>"
        f"MSE: {row['mse_mean']:.6f} ± {row['mse_std']:.6f}<br>"
        f"IC: {row['ic_mean']:.4f} ± {row['ic_std']:.4f}<br>"
        f"Seeds: {row['n_seeds']}"
    )


def _short_name(name):
    """Shorten augmenter name for annotation."""
    name = name.replace("qunified_", "q:").replace("_3L_3ens", "")
    name = name.replace("poly_deg2_interact", "poly_interact")
    name = name.replace("interaction_log", "interact+log")
    name = name.replace("poly_deg2", "poly_deg2")
    return name


# Style definitions per legend group
GROUP_STYLES = {
    "Quantum":        dict(color="#4C78A8", symbol="diamond"),
    "RFF":            dict(color="#F58518", symbol="triangle-up"),
    "Polynomial":     dict(color="#B279A2", symbol="triangle-down"),
    "Interaction+Log": dict(color="#EECA3B", symbol="hexagon"),
    "Identity":       dict(color="#9D755D", symbol="square"),
}


def plot_pareto(agg: pd.DataFrame, title_suffix: str = ""):
    """Two-panel plot: features vs MSE (left), features vs IC (right).

    Quantum and RFF Pareto frontiers are drawn as lines.
    All points are individually annotated.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    agg = agg.copy()
    agg["hover"] = agg.apply(hover_text, axis=1)
    agg["group"] = agg["augmenter_name"].apply(classify)

    # Split by group
    groups = {}
    for g in GROUP_STYLES:
        groups[g] = agg[agg["group"] == g].sort_values("n_features")

    # Compute Pareto frontiers for Quantum and RFF
    q_pareto_mse = _pareto_frontier(groups["Quantum"], "mse_mean", True) if len(groups.get("Quantum", [])) else pd.DataFrame()
    q_pareto_ic = _pareto_frontier(groups["Quantum"], "ic_mean", False) if len(groups.get("Quantum", [])) else pd.DataFrame()
    r_pareto_mse = _pareto_frontier(groups["RFF"], "mse_mean", True) if len(groups.get("RFF", [])) else pd.DataFrame()
    r_pareto_ic = _pareto_frontier(groups["RFF"], "ic_mean", False) if len(groups.get("RFF", [])) else pd.DataFrame()

    # Identity reference
    identity_row = groups.get("Identity", pd.DataFrame())
    identity_mse = identity_row["mse_mean"].iloc[0] if len(identity_row) else None
    identity_ic = identity_row["ic_mean"].iloc[0] if len(identity_row) else None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Feature Count vs OOS MSE", "Feature Count vs Information Coefficient"],
        horizontal_spacing=0.10,
    )

    def _add_pareto_line(col, pareto_df, y_col, style, name):
        """Add a Pareto frontier line (no markers, no legend — just the envelope)."""
        if pareto_df.empty:
            return
        fig.add_trace(go.Scatter(
            x=pareto_df["n_features"], y=pareto_df[y_col],
            mode="lines", name=name,
            line=dict(color=style["color"], width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

    # Precompute annotation offsets to avoid overlap.
    # Points at the same n_features get fanned out vertically.
    _annotation_offsets = {}  # (col, n_features) -> counter

    def _get_offset(col, n_features, group_name):
        """Return (ax, ay) pixel offset, fanning out stacked points."""
        key = (col, n_features)
        idx = _annotation_offsets.get(key, 0)
        _annotation_offsets[key] = idx + 1

        # Base direction per group
        if group_name in ("Quantum",):
            base_ax = 45
        elif group_name == "RFF":
            base_ax = -45
        else:
            base_ax = -50

        # Fan vertically: alternate above/below, increasing spread
        sign = 1 if idx % 2 == 0 else -1
        step = (idx // 2) + 1
        ay = sign * step * 16

        return base_ax, ay

    def _add_group_points(col, y_col, y_err_col, group_name, group_df, style, show_legend):
        """Add all points for one group with error bars and annotations."""
        if group_df.empty:
            return

        # Add legend entry once
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=group_name,
            marker=dict(size=9, **style),
            legendgroup=group_name, showlegend=show_legend,
        ), row=1, col=col)

        # Add each point
        for _, row in group_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["n_features"]], y=[row[y_col]],
                error_y=dict(type="data", array=[row[y_err_col]],
                             visible=True, thickness=1.5, width=4),
                mode="markers",
                marker=dict(size=9, **style),
                hovertext=[row["hover"]], hoverinfo="text",
                legendgroup=group_name, showlegend=False,
            ), row=1, col=col)

            ax, ay = _get_offset(col, row["n_features"], group_name)

            fig.add_annotation(
                x=np.log10(row["n_features"]), y=row[y_col],
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                text=_short_name(row["augmenter_name"]),
                showarrow=True, arrowhead=0, arrowwidth=0.5, arrowcolor="#ccc",
                ax=ax, ay=ay,
                font=dict(size=8, color=style["color"]),
            )

    # --- Panel 1: MSE ---
    _add_pareto_line(1, q_pareto_mse, "mse_mean", GROUP_STYLES["Quantum"], "Quantum Pareto")
    _add_pareto_line(1, r_pareto_mse, "mse_mean", GROUP_STYLES["RFF"], "RFF Pareto")
    for g_name in ["Quantum", "RFF", "Polynomial", "Interaction+Log", "Identity"]:
        _add_group_points(1, "mse_mean", "mse_std", g_name,
                          groups.get(g_name, pd.DataFrame()), GROUP_STYLES[g_name],
                          show_legend=True)

    if identity_mse is not None:
        fig.add_hline(y=identity_mse, line_dash="dash", line_color="#d62728",
                      line_width=1, opacity=0.4, row=1, col=1)

    # --- Panel 2: IC ---
    _add_pareto_line(2, q_pareto_ic, "ic_mean", GROUP_STYLES["Quantum"], "Quantum Pareto")
    _add_pareto_line(2, r_pareto_ic, "ic_mean", GROUP_STYLES["RFF"], "RFF Pareto")
    for g_name in ["Quantum", "RFF", "Polynomial", "Interaction+Log", "Identity"]:
        _add_group_points(2, "ic_mean", "ic_std", g_name,
                          groups.get(g_name, pd.DataFrame()), GROUP_STYLES[g_name],
                          show_legend=False)

    if identity_ic is not None:
        fig.add_hline(y=identity_ic, line_dash="dash", line_color="#d62728",
                      line_width=1, opacity=0.4, row=1, col=2)

    # Axes
    for col in (1, 2):
        fig.update_xaxes(title_text="# Features", type="log", row=1, col=col)
    fig.update_yaxes(title_text="OOS MSE", row=1, col=1)
    fig.update_yaxes(title_text="IC (Pearson r)", row=1, col=2)

    fig.update_layout(
        title=f"Real Data: Quantum vs Classical Feature Augmentation (5 seeds, Ridge){title_suffix}",
        height=650, width=1500,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )

    path = FIGURES_DIR / "pareto_vs_classical.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")

    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path.with_suffix('.png')}")
    except Exception:
        pass


def plot_mse_bar(agg: pd.DataFrame):
    """Bar chart of MSE with error bars, sorted, colored by group."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    agg = agg.sort_values("mse_mean").copy()
    agg["group"] = agg["augmenter_name"].apply(classify)
    colors = agg["group"].map({g: s["color"] for g, s in GROUP_STYLES.items()})

    fig = go.Figure(go.Bar(
        x=agg["augmenter_name"], y=agg["mse_mean"],
        error_y=dict(type="data", array=agg["mse_std"], visible=True),
        marker_color=colors,
        hovertext=agg.apply(hover_text, axis=1), hoverinfo="text",
    ))
    fig.update_layout(
        title="OOS MSE by Augmenter (5 seeds, Ridge)",
        xaxis_title="Augmenter", yaxis_title="MSE (mean ± std)",
        xaxis_tickangle=-45, template="plotly_white", height=500, width=1000,
    )
    path = FIGURES_DIR / "mse_bar.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def plot_ic_bar(agg: pd.DataFrame):
    """Bar chart of IC with error bars, sorted, colored by group."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    agg = agg.sort_values("ic_mean", ascending=False).copy()
    agg["group"] = agg["augmenter_name"].apply(classify)
    colors = agg["group"].map({g: s["color"] for g, s in GROUP_STYLES.items()})

    fig = go.Figure(go.Bar(
        x=agg["augmenter_name"], y=agg["ic_mean"],
        error_y=dict(type="data", array=agg["ic_std"], visible=True),
        marker_color=colors,
        hovertext=agg.apply(hover_text, axis=1), hoverinfo="text",
    ))
    fig.update_layout(
        title="OOS Information Coefficient by Augmenter (5 seeds, Ridge)",
        xaxis_title="Augmenter", yaxis_title="IC (mean ± std)",
        xaxis_tickangle=-45, template="plotly_white", height=500, width=1000,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    path = FIGURES_DIR / "ic_bar.html"
    fig.write_html(str(path), include_plotlyjs=True)
    print(f"Saved: {path}")


def main(run_prefix: str = "full"):
    print(f"Loading {run_prefix}_s* results across {len(SEEDS)} seeds...")
    raw = load_multi_seed(run_prefix)
    agg = aggregate(raw)

    print(f"\nAggregated results ({len(agg)} augmenters × {agg['n_seeds'].iloc[0]} seeds):")
    print(agg.sort_values("mse_mean")[
        ["augmenter_name", "mse_mean", "mse_std", "ic_mean", "ic_std", "n_features"]
    ].to_string(index=False))

    print(f"\nGenerating plots...")
    plot_pareto(agg)
    plot_mse_bar(agg)
    plot_ic_bar(agg)
    print(f"\nAll plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    run_prefix = sys.argv[1] if len(sys.argv) > 1 else "full"
    main(run_prefix)
