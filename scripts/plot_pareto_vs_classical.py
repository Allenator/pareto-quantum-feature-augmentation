"""Plot quantum Pareto frontier vs classical baselines — MSE and Correlation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = Path("results/synthetic/pareto_vs_classical")
FIGURES_DIR = Path("plots/synthetic/pareto_vs_classical")

# Reference lines
_IDENTITY_MSE = None  # computed from data
_ORACLE_MSE = None
_NOISE_FLOOR_MSE = 1.0

def _pareto_frontier(df_sub, metric_col="mse_mean", lower_is_better=True):
    df_sorted = df_sub.sort_values("n_features").reset_index(drop=True)
    frontier = []
    best = float("inf") if lower_is_better else float("-inf")
    for _, row in df_sorted.iterrows():
        val = row[metric_col]
        if (lower_is_better and val < best) or (not lower_is_better and val > best):
            best = val
            frontier.append(row)
    return pd.DataFrame(frontier)


def load_and_classify():
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")

    def classify(name):
        if name.startswith("rff_"):
            return "RFF"
        if name.startswith("unified_"):
            return "quantum"
        # identity, oracle, poly_deg2, poly_deg2_interact, interaction_log
        return name

    summary["method_type"] = summary["augmenter_name"].apply(classify)
    return summary


def make_plot():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_classify()

    # Compute correlation bounds from data
    from src.synthetic.dgp import get_or_generate
    from src.synthetic.config import DGPConfig
    _, _, _, y_te, _, _ = get_or_generate(DGPConfig(), "data/synthetic")
    corr_ceiling = np.sqrt(1 - 1 / np.var(y_te))

    identity_mse = df.loc[df["augmenter_name"] == "identity", "mse_mean"].values[0]
    oracle_mse = df.loc[df["augmenter_name"] == "oracle", "mse_mean"].values[0]
    identity_corr = df.loc[df["augmenter_name"] == "identity", "corr_mean"].values[0]
    oracle_corr = df.loc[df["augmenter_name"] == "oracle", "corr_mean"].values[0]

    # Compute overfit
    df["overfit"] = df["mse_mean"] - df["mse_train_mean"]

    def hover(row):
        return (
            f"<b>{row['augmenter_name']}</b><br>"
            f"Features: {row['n_features']}<br>"
            f"MSE: {row['mse_mean']:.3f} ± {row.get('mse_std', 0):.3f}<br>"
            f"MSE train: {row.get('mse_train_mean', 0):.3f}<br>"
            f"Overfit: {row.get('overfit', 0):.3f}<br>"
            f"Corr: {row['corr_mean']:.4f} ± {row.get('corr_std', 0):.4f}"
        )
    df["hover"] = df.apply(hover, axis=1)

    # Split data
    quantum = df[df["method_type"] == "quantum"]
    rff = df[df["method_type"] == "RFF"].sort_values("n_features")

    # Individual classical methods (each a single point)
    _CLASSICAL_ORDER = ["identity", "oracle", "interaction_log", "poly_deg2", "poly_deg2_interact"]
    individual_classical = df[~df["method_type"].isin(["quantum", "RFF"])]
    individual_classical = individual_classical.set_index("augmenter_name").loc[
        [n for n in _CLASSICAL_ORDER if n in individual_classical["augmenter_name"].values]
    ].reset_index()

    # Pareto frontiers
    q_pareto_mse = _pareto_frontier(quantum, "mse_mean", True)
    q_pareto_corr = _pareto_frontier(quantum, "corr_mean", False)
    q_pareto_mse["hover"] = q_pareto_mse.apply(hover, axis=1)
    q_pareto_corr["hover"] = q_pareto_corr.apply(hover, axis=1)

    # Classical trace styles
    _CLASSICAL_STYLES = {
        "identity":           dict(color="#9D755D", symbol="square"),
        "oracle":             dict(color="#54A24B", symbol="diamond"),
        "poly_deg2":          dict(color="#B279A2", symbol="triangle-up"),
        "poly_deg2_interact": dict(color="#B279A2", symbol="triangle-down"),
        "interaction_log":    dict(color="#EECA3B", symbol="hexagon"),
    }

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Test MSE", "Test Correlation"],
                        horizontal_spacing=0.08)

    def add_traces(col, y_col, y_err_col, q_pareto, show_legend=True):
        """Add all method traces to a panel."""
        # Quantum Pareto
        err = None
        if y_err_col and y_err_col in q_pareto.columns:
            err = dict(type="data", array=q_pareto[y_err_col].fillna(0), visible=True, thickness=1.5, width=4)
        fig.add_trace(go.Scatter(
            x=q_pareto["n_features"], y=q_pareto[y_col],
            error_y=err,
            mode="lines+markers", name="Quantum Pareto",
            marker=dict(size=9, color="#4C78A8", symbol="diamond"),
            line=dict(color="#4C78A8", width=2.5),
            hovertext=q_pareto["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # RFF line
        rff_err = None
        if y_err_col and y_err_col in rff.columns:
            rff_err = dict(type="data", array=rff[y_err_col].fillna(0), visible=True, thickness=1, width=3)
        fig.add_trace(go.Scatter(
            x=rff["n_features"], y=rff[y_col],
            error_y=rff_err,
            mode="lines+markers", name="RFF",
            marker=dict(size=7, color="#F58518", symbol="triangle-up"),
            line=dict(color="#F58518", width=2),
            hovertext=rff["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # Individual classical methods
        for _, row in individual_classical.iterrows():
            sty = _CLASSICAL_STYLES.get(row["augmenter_name"], dict(color="#72B7B2", symbol="circle"))
            pt_err = None
            if y_err_col and y_err_col in row.index:
                pt_err = dict(type="data", array=[row.get(y_err_col, 0)], visible=True, thickness=1.5, width=5)
            fig.add_trace(go.Scatter(
                x=[row["n_features"]], y=[row[y_col]],
                error_y=pt_err,
                mode="markers", name=row["augmenter_name"],
                marker=dict(size=9, **sty),
                hovertext=[row["hover"]], hoverinfo="text",
                showlegend=show_legend,
            ), row=1, col=col)

    # Panel 1: MSE
    add_traces(1, "mse_mean", "mse_std", q_pareto_mse, show_legend=True)

    # Reference lines — MSE
    for y_val, label, color in [
        (identity_mse, "Identity", "#d62728"),
        (oracle_mse, "Oracle", "#2ca02c"),
        (1.0, "Noise floor", "#888888"),
    ]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1.5, opacity=0.6, row=1, col=1)
        fig.add_annotation(x=0.98, y=y_val, xref="x domain", yref="y",
                           text=label, showarrow=False, xanchor="right", yanchor="top",
                           yshift=-3, font=dict(size=9, color=color), row=1, col=1)

    # Panel 2: Correlation
    add_traces(2, "corr_mean", "corr_std", q_pareto_corr, show_legend=False)

    # Reference lines — Correlation (panel 2)
    for y_val, label, color in [
        (identity_corr, "Identity", "#d62728"),
        (oracle_corr, "Oracle", "#2ca02c"),
        (corr_ceiling, f"Ceiling (ρ={corr_ceiling:.3f})", "#888888"),
    ]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1.5, opacity=0.6, row=1, col=2)
        fig.add_annotation(x=0.98, y=y_val, xref="x2 domain", yref="y2",
                           text=label, showarrow=False, xanchor="right", yanchor="bottom",
                           yshift=3, font=dict(size=9, color=color), row=1, col=2)

    for col in (1, 2):
        fig.update_xaxes(title_text="# Features", type="log", row=1, col=col)
    fig.update_yaxes(title_text="Test MSE", row=1, col=1)
    fig.update_yaxes(title_text="Test Correlation", row=1, col=2)

    fig.update_layout(
        title="Quantum Pareto Frontier vs Classical Baselines (10K/10K, 5 seeds, Ridge)",
        height=600, width=1400,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    path = FIGURES_DIR / "pareto_vs_classical.html"
    fig.write_html(str(path), include_plotlyjs=True)
    fig.write_image(str(path.with_suffix(".png")), scale=3)
    print(f"Saved: {path} + .png")

    # ── Overfitting diagnostic (separate plot) ──
    make_overfit_plot(df, quantum, rff, individual_classical, q_pareto_mse, _CLASSICAL_STYLES, hover)
    make_params_plot(df, quantum, rff, individual_classical, q_pareto_mse, q_pareto_corr, _CLASSICAL_STYLES, hover)


def make_overfit_plot(df, quantum, rff, individual_classical, q_pareto_mse, styles, hover_fn):
    """Separate overfitting diagnostic: train MSE, test MSE, and gap."""
    df = df.copy()
    df["overfit"] = df["mse_mean"] - df["mse_train_mean"]

    quantum = quantum.copy()
    quantum["overfit"] = quantum["mse_mean"] - quantum["mse_train_mean"]
    rff = rff.copy()
    rff["overfit"] = rff["mse_mean"] - rff["mse_train_mean"]
    individual_classical = individual_classical.copy()
    individual_classical["overfit"] = individual_classical["mse_mean"] - individual_classical["mse_train_mean"]
    q_pareto = q_pareto_mse.copy()
    q_pareto["overfit"] = q_pareto["mse_mean"] - q_pareto["mse_train_mean"]

    # Recompute hover with overfit
    for sub in [quantum, rff, individual_classical, q_pareto]:
        sub["hover"] = sub.apply(hover_fn, axis=1)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Train MSE", "Test MSE", "Overfit (Test − Train)"],
                        horizontal_spacing=0.06)

    panels = [
        (1, "mse_train_mean", "mse_train_std"),
        (2, "mse_mean", "mse_std"),
        (3, "overfit", None),
    ]

    for col, y_col, y_err_col in panels:
        show_legend = (col == 1)

        # Quantum Pareto
        err = None
        if y_err_col and y_err_col in q_pareto.columns:
            err = dict(type="data", array=q_pareto[y_err_col].fillna(0), visible=True, thickness=1.5, width=4)
        fig.add_trace(go.Scatter(
            x=q_pareto["n_features"], y=q_pareto[y_col],
            error_y=err,
            mode="lines+markers", name="Quantum Pareto",
            marker=dict(size=9, color="#4C78A8", symbol="diamond"),
            line=dict(color="#4C78A8", width=2.5),
            hovertext=q_pareto["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # RFF
        rff_err = None
        if y_err_col and y_err_col in rff.columns:
            rff_err = dict(type="data", array=rff[y_err_col].fillna(0), visible=True, thickness=1, width=3)
        fig.add_trace(go.Scatter(
            x=rff["n_features"], y=rff[y_col],
            error_y=rff_err,
            mode="lines+markers", name="RFF",
            marker=dict(size=7, color="#F58518", symbol="triangle-up"),
            line=dict(color="#F58518", width=2),
            hovertext=rff["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # Classical points
        for _, row in individual_classical.iterrows():
            sty = styles.get(row["augmenter_name"], dict(color="#72B7B2", symbol="circle"))
            pt_err = None
            if y_err_col and y_err_col in row.index:
                pt_err = dict(type="data", array=[row.get(y_err_col, 0)], visible=True, thickness=1.5, width=5)
            fig.add_trace(go.Scatter(
                x=[row["n_features"]], y=[row[y_col]],
                error_y=pt_err,
                mode="markers", name=row["augmenter_name"],
                marker=dict(size=9, **sty),
                hovertext=[row["hover"]], hoverinfo="text",
                showlegend=show_legend,
            ), row=1, col=col)

    # Reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888888", line_width=1.5, opacity=0.6, row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888888", line_width=1.5, opacity=0.6, row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1.5, opacity=0.6, row=1, col=3)
    fig.add_annotation(x=0.98, y=0, xref="x3 domain", yref="y3",
                       text="No overfitting", showarrow=False, xanchor="right", yanchor="top",
                       yshift=-3, font=dict(size=9, color="#888888"), row=1, col=3)

    for col in (1, 2, 3):
        fig.update_xaxes(title_text="# Features", type="log", row=1, col=col)
    fig.update_yaxes(title_text="Train MSE", row=1, col=1)
    fig.update_yaxes(title_text="Test MSE", row=1, col=2)
    fig.update_yaxes(title_text="Test − Train MSE", row=1, col=3)

    fig.update_layout(
        title="Overfitting Diagnostic (10K/10K, 5 seeds, Ridge)",
        height=550, width=1600,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    path = FIGURES_DIR / "overfitting_diagnostic.html"
    fig.write_html(str(path), include_plotlyjs=True)
    fig.write_image(str(path.with_suffix(".png")), scale=3)
    print(f"Saved: {path} + .png")


def make_params_plot(df, quantum, rff, individual_classical, q_pareto_mse, q_pareto_corr, styles, hover_fn):
    """MSE and Correlation vs # random parameters."""
    # Classical methods have 0 random params except RFF
    # Shift 0 to 0.5 for log-scale visibility
    for sub in [quantum, rff, individual_classical, q_pareto_mse, q_pareto_corr]:
        sub = sub.copy()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Test MSE vs Random Params", "Test Correlation vs Random Params"],
                        horizontal_spacing=0.08)

    panels = [
        (1, "mse_mean", "mse_std", q_pareto_mse),
        (2, "corr_mean", "corr_std", q_pareto_corr),
    ]

    for col, y_col, y_err_col, q_pareto in panels:
        show_legend = (col == 1)

        def x_val(v):
            """Shift 0 params to 0.5 for log scale."""
            return max(v, 0.5)

        # Quantum Pareto
        err = dict(type="data", array=q_pareto[y_err_col].fillna(0), visible=True, thickness=1.5, width=4)
        fig.add_trace(go.Scatter(
            x=q_pareto["n_random_params"].apply(x_val), y=q_pareto[y_col],
            error_y=err,
            mode="markers", name="Quantum Pareto",
            marker=dict(size=9, color="#4C78A8", symbol="diamond"),
            hovertext=q_pareto["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # RFF
        rff_err = dict(type="data", array=rff[y_err_col].fillna(0), visible=True, thickness=1, width=3)
        fig.add_trace(go.Scatter(
            x=rff["n_random_params"].apply(x_val), y=rff[y_col],
            error_y=rff_err,
            mode="markers", name="RFF",
            marker=dict(size=7, color="#F58518", symbol="triangle-up"),
            hovertext=rff["hover"], hoverinfo="text",
            showlegend=show_legend,
        ), row=1, col=col)

        # Classical points
        for _, row in individual_classical.iterrows():
            sty = styles.get(row["augmenter_name"], dict(color="#72B7B2", symbol="circle"))
            pt_err = dict(type="data", array=[row.get(y_err_col, 0)], visible=True, thickness=1.5, width=5)
            fig.add_trace(go.Scatter(
                x=[x_val(row["n_random_params"])], y=[row[y_col]],
                error_y=pt_err,
                mode="markers", name=row["augmenter_name"],
                marker=dict(size=9, **sty),
                hovertext=[row["hover"]], hoverinfo="text",
                showlegend=show_legend,
            ), row=1, col=col)

    fig.add_hline(y=1.0, line_dash="dash", line_color="#888888", line_width=1.5, opacity=0.6, row=1, col=1)

    for col in (1, 2):
        fig.update_xaxes(title_text="# Random Parameters", type="log", row=1, col=col)
    fig.update_yaxes(title_text="Test MSE", row=1, col=1)
    fig.update_yaxes(title_text="Test Correlation", row=1, col=2)

    fig.update_layout(
        title="Performance vs Model Complexity (Random Parameters)",
        height=600, width=1400,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )

    path = FIGURES_DIR / "params_vs_performance.html"
    fig.write_html(str(path), include_plotlyjs=True)
    fig.write_image(str(path.with_suffix(".png")), scale=3)
    print(f"Saved: {path} + .png")


if __name__ == "__main__":
    make_plot()
