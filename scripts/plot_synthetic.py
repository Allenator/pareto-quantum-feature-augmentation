"""Plot static sweep results: test MSE as a function of # features, per regime."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = Path("results/synthetic/static_sweep")
FIGURES_DIR = Path("plots/synthetic/static_sweep/figures")

# (family, group, prefixes, opts)
FAMILY_RULES = [
    ("Polynomial (deg 2)", "Polynomial",  ["poly_deg2_interact", "poly_deg2"], {"exclude": ["poly_deg3"]}),
    ("Polynomial (deg 3)", "Polynomial",  ["poly_deg3"], {}),
    ("RFF",                "RFF",         ["rff_"], {}),
    ("Log/Abs",            "RFF",         ["interaction_log"], {}),
    ("Angle Encoding",     "Angle/Prob",  ["angle_strong_"], {}),
    ("Probability",        "Angle/Prob",  ["prob_"], {}),
    ("ZZ Map",             "ZZ/IQP/QAOA", ["zz_"], {}),
    ("IQP",                "ZZ/IQP/QAOA", ["iqp_"], {}),
    ("QAOA",               "ZZ/IQP/QAOA", ["qaoa_"], {}),
    ("Reservoir (Z)",      "Reservoir",   ["reservoir_"], {"exclude": ["_zz", "_X", "_Y", "_full", "_circ", "_reup", "_all"]}),
    ("Reservoir (Z+ZZ)",   "Reservoir",   ["reservoir_"], {"require": ["_zz"], "exclude": ["_XYZ", "_circ", "_reup", "_all"]}),
    ("Reservoir (XYZ)",    "Reservoir",   ["reservoir_"], {"require": ["_XYZ"], "exclude": ["_ZZ"]}),
    ("Reservoir (XYZ+ZZ)", "Reservoir",   ["reservoir_"], {"require": ["_XYZ_ZZ"]}),
    ("Reservoir (full)",   "Reservoir",   ["reservoir_"], {"require": ["_full"]}),
    ("Reservoir (other)",  "Reservoir",   ["reservoir_"], {"require": ["_X", "_Y", "_circ", "_reup", "_all"]}),
    ("Oracle",             "Reference",   ["oracle"], {}),
    ("Identity",           "Reference",   ["identity"], {}),
]

GROUP_ORDER = ["Polynomial", "RFF", "Angle/Prob", "ZZ/IQP/QAOA", "Reservoir", "Reference"]

GROUP_COLORS = {
    "Polynomial":  "#1f77b4",
    "RFF":         "#ff7f0e",
    "Angle/Prob":  "#2ca02c",
    "ZZ/IQP/QAOA": "#d62728",
    "Reservoir":   "#e377c2",
    "Reference":   "#555555",
}

MARKER_POOL = [
    "circle", "square", "diamond", "triangle-up", "triangle-down",
    "cross", "x", "star", "hexagon", "pentagon",
]


def load_results() -> pd.DataFrame:
    records = []
    for p in RESULTS_DIR.rglob("*.json"):
        with open(p) as f:
            records.append(json.load(f))
    if not records:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")
    return pd.DataFrame(records)


def classify(name: str) -> tuple[str, str] | None:
    for family, group, prefixes, opts in FAMILY_RULES:
        exclude = opts.get("exclude", [])
        require = opts.get("require", [])
        for pfx in prefixes:
            if not (name.startswith(pfx) or name == pfx):
                continue
            if any(ex in name for ex in exclude):
                continue
            if require and not any(req in name for req in require):
                continue
            return family, group
    return None


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("augmenter_name").agg(
        n_features=("n_features_total", "first"),
        # Test MSE
        mse_all=("mse", "mean"), mse_all_std=("mse", "std"),
        mse_r1=("mse_regime1", "mean"), mse_r1_std=("mse_regime1", "std"),
        mse_r2=("mse_regime2", "mean"), mse_r2_std=("mse_regime2", "std"),
        # Test MAE
        mae_all=("mae", "mean"), mae_all_std=("mae", "std"),
        mae_r1=("mae_regime1", "mean"), mae_r1_std=("mae_regime1", "std"),
        mae_r2=("mae_regime2", "mean"), mae_r2_std=("mae_regime2", "std"),
        # Test Corr
        corr_all=("pearson_r", "mean"), corr_all_std=("pearson_r", "std"),
        corr_r1=("pearson_r_regime1", "mean"), corr_r1_std=("pearson_r_regime1", "std"),
        corr_r2=("pearson_r_regime2", "mean"), corr_r2_std=("pearson_r_regime2", "std"),
        # Train MSE
        mse_train_all=("mse_train", "mean"), mse_train_all_std=("mse_train", "std"),
        mse_train_r1=("mse_train_regime1", "mean"), mse_train_r1_std=("mse_train_regime1", "std"),
        mse_train_r2=("mse_train_regime2", "mean"), mse_train_r2_std=("mse_train_regime2", "std"),
        # Complexity
        n_trainable_params=("n_trainable_params", "first"),
        n_random_params=("n_random_params", "first"),
        effective_rank=("effective_rank", "mean"),
        nonlinearity_score=("nonlinearity_score", "mean"),
        feature_target_alignment=("feature_target_alignment", "mean"),
    ).reset_index()

    families, groups = [], []
    for name in agg["augmenter_name"]:
        result = classify(name)
        families.append(result[0] if result else None)
        groups.append(result[1] if result else None)
    agg["family"] = families
    agg["group"] = groups
    return agg.dropna(subset=["family"])


def build_long_df(df: pd.DataFrame, metric="mse") -> pd.DataFrame:
    """Build long-form for a given metric. Supports 'mse', 'mae', 'corr'."""
    agg = _aggregate(df)

    # Map metric to column names — (all, r1, r2) with std variants
    col_map = {
        "mse": (("mse_all", "mse_all_std"), ("mse_r1", "mse_r1_std"), ("mse_r2", "mse_r2_std")),
        "mae": (("mae_all", "mae_all_std"), ("mae_r1", "mae_r1_std"), ("mae_r2", "mae_r2_std")),
        "corr": (("corr_all", "corr_all_std"), ("corr_r1", "corr_r1_std"), ("corr_r2", "corr_r2_std")),
    }
    cols = col_map[metric]
    REGIMES = ["All Data", "Regime 1 (Linear)", "Regime 2 (Nonlinear)"]

    rows = []
    for _, r in agg.iterrows():
        base = {"augmenter": r.augmenter_name, "family": r.family,
                "group": r.group, "n_features": r.n_features}
        for regime, (val_col, std_col) in zip(REGIMES, cols):
            if val_col is None:
                continue
            rows.append({**base, "regime": regime,
                         "value": r.get(val_col), "value_std": r.get(std_col)})

    long = pd.DataFrame(rows).dropna(subset=["value"])
    long["regime"] = pd.Categorical(long["regime"],
                                     categories=[r for r in REGIMES if r in long["regime"].values],
                                     ordered=True)
    return long


def build_train_test_long_df(df: pd.DataFrame) -> pd.DataFrame:
    agg = _aggregate(df)
    REGIMES = ["All Data", "Regime 1 (Linear)", "Regime 2 (Nonlinear)"]
    test_cols = ["mse_all", "mse_r1", "mse_r2"]
    test_std_cols = ["mse_all_std", "mse_r1_std", "mse_r2_std"]
    train_cols = ["mse_train_all", "mse_train_r1", "mse_train_r2"]
    train_std_cols = ["mse_train_all_std", "mse_train_r1_std", "mse_train_r2_std"]

    rows = []
    for _, r in agg.iterrows():
        base = {"augmenter": r.augmenter_name, "family": r.family,
                "group": r.group, "n_features": r.n_features}
        for regime, tc, tsc, trc, trsc in zip(REGIMES, test_cols, test_std_cols, train_cols, train_std_cols):
            test_val = r[tc]
            train_val = r[trc]
            overfit = test_val - train_val if pd.notna(train_val) else None
            rows.append({**base, "regime": regime, "split": "Test", "mse": test_val, "mse_std": r[tsc]})
            rows.append({**base, "regime": regime, "split": "Train", "mse": train_val, "mse_std": r[trsc]})
            rows.append({**base, "regime": regime, "split": "Overfit (Test - Train)", "mse": overfit, "mse_std": None})

    long = pd.DataFrame(rows).dropna(subset=["mse"])
    long["regime"] = pd.Categorical(long["regime"], categories=REGIMES, ordered=True)
    long["split"] = pd.Categorical(
        long["split"],
        categories=["Train", "Test", "Overfit (Test - Train)"],
        ordered=True,
    )
    return long


def build_style_maps():
    color_map = {}
    symbol_map = {}
    group_marker_idx = {}
    for family, group, _, _ in FAMILY_RULES:
        color_map[family] = GROUP_COLORS.get(group, "#999999")
        idx = group_marker_idx.get(group, 0)
        symbol_map[family] = MARKER_POOL[idx % len(MARKER_POOL)]
        group_marker_idx[group] = idx + 1
    return color_map, symbol_map


def _legend_col_for_group(group: str) -> int:
    try:
        return GROUP_ORDER.index(group)
    except ValueError:
        return len(GROUP_ORDER)


def _legend_rank(family: str, group: str) -> int:
    col = _legend_col_for_group(group)
    pos = 0
    for i, (fam, grp, _, _) in enumerate(FAMILY_RULES):
        if fam == family:
            pos = i
            break
    return col * 100 + pos


def _apply_legend_groups(fig, long):
    """Apply grouped legend with bold group titles and sorted ranks."""
    family_to_group = {}
    for family in long["family"].unique():
        family_to_group[family] = long.loc[long["family"] == family, "group"].iloc[0]

    for trace in fig.data:
        family = trace.name
        group = family_to_group.get(family, "Reference")
        trace.legendgroup = group
        trace.legendgrouptitle = dict(text=f"<b>{group}</b>", font=dict(size=10))
        trace.legendrank = _legend_rank(family, group)


def _compute_bounds():
    """Compute per-regime theoretical bounds from DGP data."""
    from src.synthetic.dgp import get_or_generate
    from src.synthetic.config import DGPConfig
    _, _, _, y_te, _, r_te = get_or_generate(DGPConfig(), "data/synthetic")

    var_all = float(np.var(y_te))
    var_r1 = float(np.var(y_te[r_te == 1]))
    var_r2 = float(np.var(y_te[r_te == 2]))
    mae_floor = float(np.sqrt(2 / np.pi))

    return {
        "MSE": {
            "All Data": 1.0,
            "Regime 1 (Linear)": 1.0,
            "Regime 2 (Nonlinear)": 1.0,
            "caption": "Dashed line represents noise floor due to ε (MSE = Var(ε) = 1)",
        },
        "MAE": {
            "All Data": mae_floor,
            "Regime 1 (Linear)": mae_floor,
            "Regime 2 (Nonlinear)": mae_floor,
            "caption": f"Dashed line represents noise floor due to ε (MAE = √(2/π) ≈ {mae_floor:.3f})",
        },
        "Correlation": {
            "All Data": np.sqrt(1 - 1 / var_all),
            "Regime 1 (Linear)": np.sqrt(1 - 1 / var_r1),
            "Regime 2 (Nonlinear)": np.sqrt(1 - 1 / var_r2),
            "caption": "Dashed line represents correlation ceiling ρ_max = √(1 − Var(ε)/Var(Y))",
        },
    }


def _apply_layout(fig, title, metric_name="MSE", log_y=False, bounds=None):
    """Common layout for all figures."""
    if bounds and metric_name in bounds:
        mb = bounds[metric_name]
        # Per-regime bounds on faceted subplots
        regimes = ["All Data", "Regime 1 (Linear)", "Regime 2 (Nonlinear)"]
        for col_idx, regime in enumerate(regimes, start=1):
            val = mb.get(regime)
            if val is not None:
                fig.add_hline(y=float(val), line_dash="dot", line_color="gray", opacity=0.5,
                              row=1, col=col_idx)
        caption = mb.get("caption")
        if caption:
            fig.add_annotation(
                text=caption,
                xref="paper", yref="paper",
                x=0.5, y=-0.30, showarrow=False,
                font=dict(size=11, color="gray"), xanchor="center",
            )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=600, width=1500,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            title=dict(text=""),
            orientation="h", yanchor="top", y=-0.38,
            xanchor="center", x=0.5,
            font=dict(size=9),
            groupclick="togglegroup", tracegroupgap=3,
        ),
        margin=dict(b=280),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])
                            if "regime=" in str(a.text) else None)


def make_test_fig(long, color_map, symbol_map, metric_name="MSE", log_y=False, bounds=None):
    """1xN test metric faceted by regime."""
    title_suffix = " (log scale)" if log_y else ""
    long = long.copy()
    long["hover"] = (
        "<b>" + long["augmenter"] + "</b><br>"
        + "Family: " + long["family"] + "<br>"
        + "Features: " + long["n_features"].astype(str) + "<br>"
        + f"{metric_name}: " + long["value"].round(3).astype(str)
        + " ± " + long["value_std"].round(3).astype(str)
    )

    fig = px.line(
        long.sort_values(["family", "n_features"]),
        x="n_features", y="value", error_y="value_std",
        color="family", symbol="family",
        facet_col="regime", markers=True,
        custom_data=["hover"], log_y=log_y,
        color_discrete_map=color_map, symbol_map=symbol_map,
        labels={"n_features": "# Features", "value": metric_name, "family": "Method"},
        title=f"Static Augmenter Sweep — Test {metric_name} vs Feature Count{title_suffix}",
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>", marker_size=8)
    _apply_legend_groups(fig, long)
    _apply_layout(fig, f"Static Augmenter Sweep — Test {metric_name} vs Feature Count{title_suffix}", metric_name, log_y, bounds)
    return fig


def make_train_test_fig(long_tt, color_map, symbol_map):
    """3x3 panel: rows=Train/Test/Overfit, cols=All/R1/R2."""
    SPLITS = ["Train", "Test", "Overfit (Test - Train)"]
    REGIMES = ["All Data", "Regime 1 (Linear)", "Regime 2 (Nonlinear)"]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"{s} — {r}" for s in SPLITS for r in REGIMES],
        horizontal_spacing=0.05, vertical_spacing=0.08,
    )

    legend_added = set()
    for row_idx, split in enumerate(SPLITS, start=1):
        for col_idx, regime in enumerate(REGIMES, start=1):
            sub = long_tt[(long_tt["split"] == split) & (long_tt["regime"] == regime)]
            for family, group, _, _ in FAMILY_RULES:
                fsub = sub[sub["family"] == family].sort_values("n_features")
                if fsub.empty:
                    continue
                show_legend = family not in legend_added
                legend_added.add(family)
                color = color_map.get(family, "#999")
                symbol = symbol_map.get(family, "circle")
                group_name = fsub["group"].iloc[0]

                hover = [
                    f"<b>{r.augmenter}</b><br>Family: {family}<br>"
                    f"Features: {int(r.n_features)}<br>MSE: {r.mse:.3f}"
                    + (f" ± {r.mse_std:.3f}" if pd.notna(r.mse_std) else "")
                    for _, r in fsub.iterrows()
                ]
                error_y = None
                if fsub["mse_std"].notna().any():
                    error_y = dict(type="data", array=fsub["mse_std"].fillna(0), visible=True, thickness=1)

                fig.add_trace(go.Scatter(
                    x=fsub["n_features"], y=fsub["mse"], error_y=error_y,
                    mode="lines+markers", name=family,
                    legendgroup=group_name,
                    legendgrouptitle=dict(text=f"<b>{group_name}</b>", font=dict(size=10)),
                    legendrank=_legend_rank(family, group_name),
                    showlegend=show_legend,
                    marker=dict(symbol=symbol, size=6, color=color),
                    line=dict(color=color, width=1.5),
                    hovertext=hover, hoverinfo="text",
                ), row=row_idx, col=col_idx)

    for row_idx in (1, 2):
        for col_idx in (1, 2, 3):
            fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.4, row=row_idx, col=col_idx)
    for col_idx in (1, 2, 3):
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray", opacity=0.4, row=3, col=col_idx)

    fig.update_layout(
        title=dict(text="Train / Test / Overfit — MSE vs Feature Count", font=dict(size=16)),
        height=1000, width=1500, template="plotly_white", hovermode="closest",
        legend=dict(title=dict(text=""), orientation="h", yanchor="top", y=-0.12,
                    xanchor="center", x=0.5, font=dict(size=9),
                    groupclick="togglegroup", tracegroupgap=3),
        margin=dict(b=200),
    )
    fig.for_each_annotation(lambda a: a.update(font_size=11))
    return fig


def make_complexity_fig(df):
    """Bubble chart: MSE vs features, size=total params, color=trainable fraction."""
    agg = _aggregate(df)
    agg["total_params"] = agg["n_trainable_params"].fillna(0) + agg["n_random_params"].fillna(0)
    agg["trainable_frac"] = agg["n_trainable_params"].fillna(0) / (agg["total_params"] + 1)
    agg["bubble_size"] = np.log1p(agg["total_params"]) * 3 + 4

    agg["hover"] = (
        "<b>" + agg["augmenter_name"] + "</b><br>"
        + "Family: " + agg["family"] + "<br>"
        + "Features: " + agg["n_features"].astype(str) + "<br>"
        + "MSE: " + agg["mse_all"].round(3).astype(str) + "<br>"
        + "Trainable: " + agg["n_trainable_params"].astype(int).astype(str) + "<br>"
        + "Random: " + agg["n_random_params"].astype(int).astype(str) + "<br>"
        + "Eff. rank: " + agg["effective_rank"].round(1).astype(str) + "<br>"
        + "Nonlinearity: " + agg["nonlinearity_score"].round(3).astype(str) + "<br>"
        + "Target align: " + agg["feature_target_alignment"].round(3).astype(str)
    )

    fig = px.scatter(
        agg, x="n_features", y="mse_all",
        size="bubble_size", color="family",
        custom_data=["hover"],
        color_discrete_map=build_style_maps()[0],
        labels={"n_features": "# Features", "mse_all": "Test MSE", "family": "Method"},
        title="Complexity Bubble Chart — MSE vs Features (size = log params)",
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=600, width=1000, template="plotly_white", hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    color_map, symbol_map = build_style_maps()
    bounds = _compute_bounds()

    # Test metric plots (MSE, MAE, Correlation)
    for metric, label in [("mse", "MSE"), ("mae", "MAE"), ("corr", "Correlation")]:
        long = build_long_df(df, metric=metric)
        for log_y, suffix in [(False, ""), (True, "_log")]:
            fig = make_test_fig(long, color_map, symbol_map, metric_name=label, log_y=log_y, bounds=bounds)
            path = FIGURES_DIR / f"test_{metric}_vs_features{suffix}.html"
            fig.write_html(str(path), include_plotlyjs=True)
            print(f"Saved: {path}")

    # Train / Test / Overfit 3x3 panel
    long_tt = build_train_test_long_df(df)
    fig_tt = make_train_test_fig(long_tt, color_map, symbol_map)
    path_tt = FIGURES_DIR / "train_test_overfit.html"
    fig_tt.write_html(str(path_tt), include_plotlyjs=True)
    print(f"Saved: {path_tt}")

    # Complexity bubble chart
    fig_cx = make_complexity_fig(df)
    path_cx = FIGURES_DIR / "complexity_bubble.html"
    fig_cx.write_html(str(path_cx), include_plotlyjs=True)
    print(f"Saved: {path_cx}")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
