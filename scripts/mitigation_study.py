"""Mitigation study: can any classical post-processing recover signal
from Rigetti Ankaa-3 features that Ridge (with full regularisation) misses?

Strategies compared, per (source, dataset size):

  raw_only              : Ridge on the 4 StandardScaled raw regressors only
                          (floor — no quantum features at all)
  exact_ideal           : Ridge on raw + exact-sim quantum features
                          (ceiling — what the noise-free reservoir gives)
  hw_baseline           : Ridge on raw + all hardware quantum features
                          (current-state reported result)
  hw_quantum_only       : Ridge on hardware quantum only, no raw
                          (probes whether Ridge was ignoring quantum)
  hw_damping_corrected  : Ridge on raw + (hw_quantum / λ), using the λ
                          from fit_noise_model.py for each source
                          (undoes multiplicative bias)
  hw_top{k}             : Ridge on raw + top-k hardware quantum features,
                          k ∈ {1, 2, 4, 8, all}. Ranking is by per-feature
                          Pearson correlation with the exact simulator
                          computed on the TRAIN split only (no test leakage).

Outputs
-------
  results/synthetic_hw/mitigation.csv  — one row per (source, size, strategy)
  plots/synthetic_hw/mitigation.{html,png}  — bar chart

Usage
-----
    uv run python scripts/mitigation_study.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go

from src.synthetic.config import DGPConfig
from src.synthetic.dgp import get_or_generate
from src.synthetic.models.linear import RidgeModel

FEATURES_DIR = Path("features/synthetic_hw")
RESULTS_DIR = Path("results/synthetic_hw")
FIGURES_DIR = Path("plots/synthetic_hw")
N_ORIG = 4
SEED = 42

# λ values from fit_noise_model.py (hardcoded to avoid script dependency;
# see results/synthetic_hw/noise_model_fit.csv for the canonical numbers).
LAMBDA = {
    "singleton_z+zz_n100+50": 0.2065,
    "packed_z_n100+50":       0.2575,  # pooled fit λ used for all sizes
    "packed_z_n500+250":      0.2575,
    "packed_z_n1000+500":     0.2575,
}

# (label, n_train, n_test, sim_path_rel, hw_path_rel)
SOURCES = [
    ("singleton_z+zz_n100+50", 100, 50,
     "lightning/seed_42/reservoir_3x3_Z+ZZ_n100+50_exact.npz",
     "rigetti/seed_42/reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz"),
    ("packed_z_n100+50", 100, 50,
     "lightning/seed_42/reservoir_3x3_Z_n100+50_exact.npz",
     "rigetti/seed_42/reservoir_3x3_Z_n100+50_s1000_packed.npz"),
    ("packed_z_n500+250", 500, 250,
     "lightning/seed_42/reservoir_3x3_Z_n500+250_exact.npz",
     "rigetti/seed_42/reservoir_3x3_Z_n500+250_s1000_packed.npz"),
    ("packed_z_n1000+500", 1000, 500,
     "lightning/seed_42/reservoir_3x3_Z_n1000+500_exact.npz",
     "rigetti/seed_42/reservoir_3x3_Z_n1000+500_s1000_packed.npz"),
]

TOP_K_VALUES = [1, 2, 4, 8]


def load_pair(sim_rel: str, hw_rel: str):
    sim = np.load(FEATURES_DIR / sim_rel)
    hw = np.load(FEATURES_DIR / hw_rel)
    return sim, hw


def ridge_mse(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> float:
    model = RidgeModel()
    pred = model.fit_predict(X_train, y_train, X_test)
    return float(np.mean((pred.y_pred - y_test) ** 2))


def rank_features_by_train_corr(sim_q_train: np.ndarray,
                                 hw_q_train: np.ndarray) -> np.ndarray:
    """Return feature indices ordered descending by |Pearson correlation|
    between sim and hardware on the TRAIN split."""
    n_feat = sim_q_train.shape[1]
    corrs = np.zeros(n_feat)
    for i in range(n_feat):
        if sim_q_train[:, i].std() > 1e-10 and hw_q_train[:, i].std() > 1e-10:
            corrs[i] = np.corrcoef(sim_q_train[:, i], hw_q_train[:, i])[0, 1]
        else:
            corrs[i] = 0.0
    # Descending by |corr|
    return np.argsort(-np.abs(corrs)), corrs


def run_source(label: str, n_train: int, n_test: int,
                sim_rel: str, hw_rel: str) -> list[dict]:
    """Run every strategy for one (source, size) and return rows for CSV."""
    sim, hw = load_pair(sim_rel, hw_rel)

    raw_train = sim["train"][:, :N_ORIG]
    raw_test = sim["test"][:, :N_ORIG]
    sim_q_train = sim["train"][:, N_ORIG:]
    sim_q_test = sim["test"][:, N_ORIG:]
    hw_q_train = hw["train"][:, N_ORIG:]
    hw_q_test = hw["test"][:, N_ORIG:]

    dgp = DGPConfig(n_train=n_train, n_test=n_test, seed=SEED)
    data = get_or_generate(dgp, "data/synthetic")
    y_train, y_test = data[2], data[3]

    lam = LAMBDA[label]
    n_q = hw_q_train.shape[1]

    rows = []

    def add(strategy: str, train_X: np.ndarray, test_X: np.ndarray,
            k: int | None = None):
        mse = ridge_mse(train_X, y_train, test_X, y_test)
        rows.append({
            "source": label,
            "n_train": n_train,
            "n_test": n_test,
            "strategy": strategy,
            "n_features": train_X.shape[1],
            "k": k if k is not None else "",
            "ridge_mse": round(mse, 4),
        })

    # 1. raw_only
    add("raw_only", raw_train, raw_test)

    # 2. exact_ideal
    add("exact_ideal", sim["train"], sim["test"])

    # 3. hw_baseline
    add("hw_baseline", hw["train"], hw["test"])

    # 4. hw_quantum_only
    add("hw_quantum_only", hw_q_train, hw_q_test)

    # 5. hw_damping_corrected (raw + hw_q / λ)
    hw_q_train_corr = hw_q_train / lam
    hw_q_test_corr = hw_q_test / lam
    add("hw_damping_corrected",
        np.hstack([raw_train, hw_q_train_corr]),
        np.hstack([raw_test, hw_q_test_corr]))

    # 6. hw_top{k} — rank by train-only correlation to avoid test leakage
    rank, corrs = rank_features_by_train_corr(sim_q_train, hw_q_train)
    for k in TOP_K_VALUES:
        if k >= n_q:
            continue
        selected = rank[:k]
        add(f"hw_top{k}",
            np.hstack([raw_train, hw_q_train[:, selected]]),
            np.hstack([raw_test, hw_q_test[:, selected]]),
            k=k)
    # Also the "all" version (just relabels hw_baseline for clarity)
    add("hw_topall",
        np.hstack([raw_train, hw_q_train]),
        np.hstack([raw_test, hw_q_test]),
        k=n_q)

    return rows


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "n_train", "n_test", "strategy",
              "n_features", "k", "ridge_mse"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot_mitigation(rows: list[dict]):
    """Grouped bar chart: one group per source, bars for each strategy.

    The `hw_quantum_only` strategy blows up (Ridge has nothing to regress
    against), so it is excluded from the bar group and its MSE is displayed
    as a text annotation above each source's group for context.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    strategy_order = [
        "raw_only", "exact_ideal", "hw_baseline",
        "hw_damping_corrected",
        "hw_top1", "hw_top2", "hw_top4", "hw_top8", "hw_topall",
    ]
    strategy_colors = {
        "raw_only":              "#9D755D",  # brown (floor)
        "exact_ideal":           "#4C78A8",  # blue (ceiling)
        "hw_baseline":           "#E45756",  # coral (current hw)
        "hw_damping_corrected":  "#B279A2",  # mauve
        "hw_top1":               "#D0D4C2",
        "hw_top2":               "#B8C29A",
        "hw_top4":               "#91A972",
        "hw_top8":               "#6E8F4E",
        "hw_topall":             "#54A24B",  # green
    }

    sources = []
    for r in rows:
        if r["source"] not in sources:
            sources.append(r["source"])

    fig = go.Figure()
    for strategy in strategy_order:
        ys = []
        xs = []
        for source in sources:
            match = [r for r in rows
                     if r["source"] == source and r["strategy"] == strategy]
            if match:
                ys.append(match[0]["ridge_mse"])
                xs.append(source)
        if not ys:
            continue
        fig.add_trace(go.Bar(
            x=xs, y=ys, name=strategy,
            marker=dict(color=strategy_colors.get(strategy, "#888")),
            text=[f"{v:.2f}" for v in ys],
            textposition="outside",
            textfont=dict(size=9),
        ))

    # Compute y-axis range from the strategies we kept (auto-calibrate)
    kept_mse = [r["ridge_mse"] for r in rows
                if r["strategy"] in strategy_order]
    y_max = max(kept_mse) * 1.15

    # Annotate quantum-only values above each source group
    for source in sources:
        match = [r for r in rows
                 if r["source"] == source and r["strategy"] == "hw_quantum_only"]
        if match:
            mse_q = match[0]["ridge_mse"]
            fig.add_annotation(
                x=source, y=y_max * 0.95,
                text=f"<i>hw_quantum_only = {mse_q:.1f}</i>",
                showarrow=False, xanchor="center", yanchor="top",
                font=dict(size=10, color="#888"),
            )

    fig.update_yaxes(range=[0, y_max])
    fig.update_layout(
        barmode="group", bargap=0.15, bargroupgap=0.02,
        title="Mitigation study — Ridge test MSE per strategy",
        xaxis_title="Source (hardware run and data size)",
        yaxis_title="Ridge test MSE (lower is better)",
        height=620, width=1500,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.01),
    )
    path = FIGURES_DIR / "mitigation.html"
    fig.write_html(str(path), include_plotlyjs=True)
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path} + .png")
    except Exception as e:
        print(f"Saved: {path}  (PNG export failed: {e})")


def print_table(rows: list[dict]):
    """Pretty table of MSE values — one row per strategy, one col per source."""
    sources = []
    strategies = []
    for r in rows:
        if r["source"] not in sources:
            sources.append(r["source"])
        if r["strategy"] not in strategies:
            strategies.append(r["strategy"])

    print(f"\n{'=' * 108}")
    print("MITIGATION STUDY — Ridge test MSE  (exact_ideal = ceiling, raw_only = floor)")
    print(f"{'=' * 108}")
    header = f"  {'strategy':<24s}"
    for s in sources:
        header += f" {s[:22]:>22s}"
    print(header)
    print(f"  {'-' * 24}" + " " + " ".join("-" * 22 for _ in sources))
    for strat in strategies:
        line = f"  {strat:<24s}"
        for src in sources:
            match = [r for r in rows
                     if r["source"] == src and r["strategy"] == strat]
            if match:
                line += f" {match[0]['ridge_mse']:>22.4f}"
            else:
                line += f" {'—':>22s}"
        print(line)
    print(f"{'=' * 108}\n")


def main():
    all_rows = []
    for label, n_tr, n_te, sim_rel, hw_rel in SOURCES:
        all_rows.extend(run_source(label, n_tr, n_te, sim_rel, hw_rel))
    print_table(all_rows)
    write_csv(all_rows, RESULTS_DIR / "mitigation.csv")
    plot_mitigation(all_rows)


if __name__ == "__main__":
    main()
