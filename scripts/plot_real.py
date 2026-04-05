"""Generate interactive Plotly plots for real data backtest results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.real.backtest import _safe_pearsonr


def load_results(results_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load summary CSV and all prediction parquets."""
    rdir = Path(results_dir)
    summary = pd.read_csv(rdir / "summary.csv")

    predictions = {}
    for pf in rdir.glob("*_predictions.parquet"):
        key = pf.stem.replace("_predictions", "")
        predictions[key] = pd.read_parquet(pf)
        predictions[key]["date"] = pd.to_datetime(predictions[key]["date"])

    return summary, predictions


def plot_mse_bar(summary: pd.DataFrame, output_dir: Path):
    """Bar chart: MSE by augmenter, grouped by model."""
    fig = px.bar(
        summary, x="augmenter_name", y="mse", color="model_name",
        barmode="group", title="Out-of-Sample MSE by Augmenter",
        labels={"mse": "OOS MSE", "augmenter_name": "Augmenter", "model_name": "Model"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(output_dir / "mse_bar.html")
    print(f"  Saved mse_bar.html")


def plot_ic_bar(summary: pd.DataFrame, output_dir: Path):
    """Bar chart: Information Coefficient (Pearson r) by augmenter."""
    fig = px.bar(
        summary, x="augmenter_name", y="pearson_r", color="model_name",
        barmode="group", title="Out-of-Sample Information Coefficient by Augmenter",
        labels={"pearson_r": "IC (Pearson r)", "augmenter_name": "Augmenter", "model_name": "Model"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(output_dir / "ic_bar.html")
    print(f"  Saved ic_bar.html")


def plot_feature_vs_mse(summary: pd.DataFrame, output_dir: Path):
    """Scatter: feature count vs MSE."""
    if "n_features_total" not in summary.columns:
        return

    fig = px.scatter(
        summary, x="n_features_total", y="mse",
        color="model_name", hover_data=["augmenter_name"],
        title="Feature Count vs OOS MSE",
        labels={"n_features_total": "Total Features", "mse": "OOS MSE"},
    )
    fig.write_html(output_dir / "feature_vs_mse.html")
    print(f"  Saved feature_vs_mse.html")


def plot_rolling_ic(predictions: dict[str, pd.DataFrame], output_dir: Path,
                    window_days: int = 63):
    """Line chart: rolling 3-month IC over time for each (augmenter, model)."""
    fig = go.Figure()

    for key, df in sorted(predictions.items()):
        df_sorted = df.sort_values("date")
        dates = sorted(df_sorted["date"].unique())

        ic_dates = []
        ic_values = []
        for i in range(window_days, len(dates)):
            window_dates = dates[i - window_days:i]
            window_df = df_sorted[df_sorted["date"].isin(window_dates)]
            if len(window_df) >= 10:
                ic = _safe_pearsonr(window_df["y_true"].values, window_df["y_pred"].values)
                ic_dates.append(dates[i])
                ic_values.append(ic)

        if ic_dates:
            fig.add_trace(go.Scatter(
                x=ic_dates, y=ic_values, name=key, mode="lines",
            ))

    fig.update_layout(
        title="Rolling 3-Month Information Coefficient",
        xaxis_title="Date",
        yaxis_title="IC (Pearson r)",
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.write_html(output_dir / "rolling_ic.html")
    print(f"  Saved rolling_ic.html")


def plot_cumulative_predictions(predictions: dict[str, pd.DataFrame], output_dir: Path):
    """Line chart: cumulative sum of (y_pred * y_true) — simulates strategy PnL direction."""
    fig = go.Figure()

    for key, df in sorted(predictions.items()):
        df_sorted = df.sort_values("date")
        # Daily aggregate: mean prediction * mean actual across tickers
        daily = df_sorted.groupby("date").agg(
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
        ).reset_index()
        daily["signal_pnl"] = daily["y_pred_mean"] * daily["y_true_mean"]
        daily["cum_pnl"] = daily["signal_pnl"].cumsum()

        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cum_pnl"], name=key, mode="lines",
        ))

    fig.update_layout(
        title="Cumulative Signal PnL (pred * actual, aggregated daily)",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.write_html(output_dir / "cumulative_pnl.html")
    print(f"  Saved cumulative_pnl.html")


def main(run_id: str = "full"):
    results_dir = f"results/real/{run_id}"
    output_dir = Path("plots/real")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    summary, predictions = load_results(results_dir)

    print(f"Generating plots...")
    plot_mse_bar(summary, output_dir)
    plot_ic_bar(summary, output_dir)
    plot_feature_vs_mse(summary, output_dir)

    if predictions:
        plot_rolling_ic(predictions, output_dir)
        plot_cumulative_predictions(predictions, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "full"
    main(run_id)
