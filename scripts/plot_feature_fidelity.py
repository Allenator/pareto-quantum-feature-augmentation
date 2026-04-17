"""Plot feature-level fidelity between simulation and Rigetti Ankaa-3.

Produces five figures (HTML + PNG) under plots/synthetic_hw/:

  singleton_vs_sim_scatter.{html,png}
    Single-panel pooled scatter on n100+50 Z+ZZ, comparing the exact simulator
    (ReservoirAugmenter circuit, no final Rot) to the 4-qubit-density
    (singleton) Rigetti run. 150 samples × 30 features = 4500 points.

  packed_vs_sim_scatter.{html,png}
    1×3 panel pooled scatter of exact sim (QuantumReservoir circuit with
    final Rot) vs packed Rigetti (20 slots × 4q = 80 qubits active per task)
    for Z at training sizes 100+50, 500+250, 1000+500.

  three_way_overlay.{html,png}
    Overlay of singleton and packed clouds on shared hardware-vs-exact axes.
    Each point is (feature_exact, feature_hardware) from its respective
    circuit family. The two clouds centre on different x regions because
    the underlying circuits differ (ReservoirAugmenter vs QuantumReservoir)
    and the observables differ (Z+ZZ vs Z), but both *should* hug y = x and
    the tightness comparison gives a qualitative 4q-vs-80q-density read.

  error_hist.{html,png}
    Overlaid |exact − hardware| histograms for singleton, packed n100+50, and
    packed n1000+500. Also overlays SV1 shot-noise-only histogram as a noise
    floor reference. Shows how much of the hardware error budget is
    shot-noise vs gate/readout/crosstalk.

  per_feature_corr.{html,png}
    Per-feature exact-sim / packed-Rigetti correlation (12 augmented features
    for the Z observable), plotted across the three data sizes. Reveals which
    qubits/reservoirs preserve signal best.

Usage:
    uv run python scripts/plot_feature_fidelity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FEATURES_DIR = Path("features/synthetic_hw")
FIGURES_DIR = Path("plots/synthetic_hw")
N_ORIG = 4

COLORS = {
    "exact":     "#4C78A8",   # steel blue — ground truth / exact simulator
    "sv1":       "#54A24B",   # green — shot-noise-only anchor
    "singleton": "#B279A2",   # mauve — 4-qubit-density Rigetti (unpacked)
    "packed":    "#E45756",   # coral — 80-qubit-density Rigetti (packed)
}


def load_pooled_aug(path: Path) -> np.ndarray:
    """Return train + test augmented-only values stacked as (n_samples, n_aug)."""
    d = np.load(path)
    return np.vstack([d["train"][:, N_ORIG:], d["test"][:, N_ORIG:]])


def save_fig(fig: go.Figure, path: Path):
    fig.write_html(str(path), include_plotlyjs=True)
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path} + .png")
    except Exception as e:
        print(f"Saved: {path}  (PNG export failed: {e})")


# ── Singleton vs sim (n100+50 Z+ZZ) ──────────────────────────────────────
def plot_singleton_vs_sim_scatter():
    exact = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                            / "reservoir_3x3_Z+ZZ_n100+50_exact.npz")
    singleton = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                                / "reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz")

    x = exact.flatten()
    y = singleton.flatten()
    corr = float(np.corrcoef(x, y)[0, 1])

    fig = go.Figure()
    lim = 1.05
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="#888888", dash="dash", width=1),
        name="y = x (perfect agreement)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=3, color=COLORS["singleton"],
                    opacity=max(0.05, 0.6 * 300 / len(x)),
                    line=dict(color="white", width=0)),
        name=f"Singleton (4q-density), r = {corr:.3f}",
    ))
    fig.update_xaxes(title_text="Exact simulator ⟨Z⟩ / ⟨ZZ⟩", range=[-lim, lim])
    fig.update_yaxes(title_text="Rigetti Ankaa-3 (singleton, 4q density)",
                     range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=("Singleton hardware vs exact simulator — Z+ZZ, n=100+50 "
               f"({len(x)} measurements)"),
        height=620, width=780,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
    )
    save_fig(fig, FIGURES_DIR / "singleton_vs_sim_scatter.html")


# ── Packed vs sim (Z, 3 sizes) ────────────────────────────────────────────
SIZE_PAIRS = [(100, 50), (500, 250), (1000, 500)]


def plot_packed_vs_sim_scatter():
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"n = {n_tr}+{n_te}" for n_tr, n_te in SIZE_PAIRS],
        horizontal_spacing=0.07,
    )
    lim = 1.05
    for col, (n_tr, n_te) in enumerate(SIZE_PAIRS, start=1):
        exact = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                                / f"reservoir_3x3_Z_n{n_tr}+{n_te}_exact.npz")
        packed = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                                 / f"reservoir_3x3_Z_n{n_tr}+{n_te}_s1000_packed.npz")
        x = exact.flatten()
        y = packed.flatten()
        corr = float(np.corrcoef(x, y)[0, 1])

        fig.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim], mode="lines",
            line=dict(color="#888888", dash="dash", width=1),
            showlegend=(col == 1), name="y = x",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4, color=COLORS["packed"],
                        opacity=max(0.08, 0.6 * 300 / len(x)),
                        line=dict(color="white", width=0)),
            showlegend=(col == 1), name="Packed (80q-density)",
        ), row=1, col=col)
        fig.add_annotation(
            xref=f"x{'' if col == 1 else col}",
            yref=f"y{'' if col == 1 else col}",
            x=-0.95, y=0.95,
            text=f"r = {corr:.3f}  ({len(x)} pts)",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=11, color="#333"),
        )
        fig.update_xaxes(title_text="Exact simulator ⟨Z⟩",
                         range=[-lim, lim], row=1, col=col)
        fig.update_yaxes(title_text="Rigetti packed" if col == 1 else None,
                         range=[-lim, lim], row=1, col=col,
                         scaleanchor=f"x{'' if col == 1 else col}",
                         scaleratio=1)
    fig.update_layout(
        title="Packed hardware vs exact simulator — Z, 12 aug feats",
        height=560, width=1500,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="center", x=0.5),
    )
    save_fig(fig, FIGURES_DIR / "packed_vs_sim_scatter.html")


# ── Three-way overlay (singleton + packed on shared axes) ────────────────
def plot_three_way_overlay():
    """Overlay 4q singleton (RA / Z+ZZ) and 80q packed (QR / Z) clouds on
    the same exact-vs-hardware axes.

    Caveats:
      * Different circuit families (ReservoirAugmenter without final Rot
        vs QuantumReservoir with final Rot) → the two clouds sample
        different subregions of the x-axis.
      * Different observable mixes (Z+ZZ vs Z) → pairwise-ZZ measurements
        may have different shot-noise properties than single-Z.
      * Same input samples (n=100+50, seed 42), same shot count (1000),
        same QPU (Ankaa-3). Active-qubit density is 4 vs 80.
    """
    ex_s = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                           / "reservoir_3x3_Z+ZZ_n100+50_exact.npz")
    singleton = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                                / "reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz")
    ex_p = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                           / "reservoir_3x3_Z_n100+50_exact.npz")
    packed = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                             / "reservoir_3x3_Z_n100+50_s1000_packed.npz")

    xs_sing = ex_s.flatten()
    ys_sing = singleton.flatten()
    xs_pk = ex_p.flatten()
    ys_pk = packed.flatten()
    corr_sing = float(np.corrcoef(xs_sing, ys_sing)[0, 1])
    corr_pk = float(np.corrcoef(xs_pk, ys_pk)[0, 1])
    err_sing = float(np.abs(xs_sing - ys_sing).mean())
    err_pk = float(np.abs(xs_pk - ys_pk).mean())

    fig = go.Figure()
    lim = 1.05
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="#888888", dash="dash", width=1),
        name="y = x",
    ))
    fig.add_trace(go.Scatter(
        x=xs_sing, y=ys_sing, mode="markers",
        marker=dict(size=3, color=COLORS["singleton"],
                    opacity=0.25, line=dict(color="white", width=0)),
        name=(f"Singleton 4q · Z+ZZ (RA)  "
              f"r={corr_sing:.3f}  mean|err|={err_sing:.3f}"),
    ))
    fig.add_trace(go.Scatter(
        x=xs_pk, y=ys_pk, mode="markers",
        marker=dict(size=3, color=COLORS["packed"],
                    opacity=0.45, line=dict(color="white", width=0)),
        name=(f"Packed 80q · Z (QR)  "
              f"r={corr_pk:.3f}  mean|err|={err_pk:.3f}"),
    ))

    fig.update_xaxes(title_text="Exact-sim expectation (own circuit family)",
                     range=[-lim, lim])
    fig.update_yaxes(title_text="Rigetti Ankaa-3 estimate",
                     range=[-lim, lim], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=("Three-way overlay: hardware fidelity at 4q vs 80q density "
               "(n=100+50, 1000 shots, Ankaa-3)"),
        height=620, width=820,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
        annotations=[
            dict(
                x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(size=10, color="#666"),
                text=("⚠ Circuit family and observable differ between the two clouds — "
                      "not a controlled crosstalk test."),
            ),
        ],
    )
    save_fig(fig, FIGURES_DIR / "three_way_overlay.html")


# ── Error histogram: noise floor comparison ───────────────────────────────
def plot_error_hist():
    # SV1 (shot-noise only) — different augmenter (5x3 XYZ+ZZ) but we care
    # about the shot-noise distribution shape, not the observable identity.
    ex_sv1 = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                             / "reservoir_5x3_XYZ+ZZ_n100+50_exact.npz")
    sv1 = load_pooled_aug(FEATURES_DIR / "sv1" / "seed_42"
                          / "reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz")

    # Singleton Z+ZZ n100+50 (4q-density Rigetti)
    ex_sing = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                              / "reservoir_3x3_Z+ZZ_n100+50_exact.npz")
    sing = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                           / "reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz")

    # Packed Z n100+50 and n1000+500 (80q-density Rigetti)
    ex_p100 = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                              / "reservoir_3x3_Z_n100+50_exact.npz")
    p100 = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                           / "reservoir_3x3_Z_n100+50_s1000_packed.npz")
    ex_p1k = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                             / "reservoir_3x3_Z_n1000+500_exact.npz")
    p1k = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                          / "reservoir_3x3_Z_n1000+500_s1000_packed.npz")

    err_sv1 = np.abs(ex_sv1 - sv1).flatten()
    err_sing = np.abs(ex_sing - sing).flatten()
    err_p100 = np.abs(ex_p100 - p100).flatten()
    err_p1k = np.abs(ex_p1k - p1k).flatten()

    bins = dict(start=0.0, end=1.0, size=0.025)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=err_sv1, histnorm="probability density",
        name=f"SV1 (shot only)       mean = {err_sv1.mean():.3f}",
        marker=dict(color=COLORS["sv1"]), opacity=0.65, xbins=bins,
    ))
    fig.add_trace(go.Histogram(
        x=err_sing, histnorm="probability density",
        name=f"Singleton 4q n100+50   mean = {err_sing.mean():.3f}",
        marker=dict(color=COLORS["singleton"]), opacity=0.55, xbins=bins,
    ))
    fig.add_trace(go.Histogram(
        x=err_p100, histnorm="probability density",
        name=f"Packed 80q n100+50    mean = {err_p100.mean():.3f}",
        marker=dict(color=COLORS["packed"]), opacity=0.55, xbins=bins,
    ))
    fig.add_trace(go.Histogram(
        x=err_p1k, histnorm="probability density",
        name=f"Packed 80q n1000+500  mean = {err_p1k.mean():.3f}",
        marker=dict(color="#9D755D"), opacity=0.55, xbins=bins,
    ))
    fig.add_vline(x=0.0303, line_dash="dash", line_color="#444",
                  line_width=1.5, opacity=0.8)
    fig.add_annotation(x=0.0303, y=1.0, xref="x", yref="paper",
                       text="shot-noise floor (σ≈0.030)",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       xshift=4, font=dict(size=10, color="#444"))

    fig.update_layout(
        title="Per-measurement error distribution (exact vs source)",
        xaxis_title="|exact − source| per measurement",
        yaxis_title="Probability density",
        barmode="overlay", bargap=0.05,
        height=540, width=1050,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5),
    )
    save_fig(fig, FIGURES_DIR / "error_hist.html")


# ── Per-feature correlation across sizes ──────────────────────────────────
def plot_per_feature_corr():
    corrs = []
    for n_tr, n_te in SIZE_PAIRS:
        exact = load_pooled_aug(FEATURES_DIR / "lightning" / "seed_42"
                                / f"reservoir_3x3_Z_n{n_tr}+{n_te}_exact.npz")
        packed = load_pooled_aug(FEATURES_DIR / "rigetti" / "seed_42"
                                 / f"reservoir_3x3_Z_n{n_tr}+{n_te}_s1000_packed.npz")
        feat_corrs = []
        for i in range(exact.shape[1]):
            if exact[:, i].std() > 1e-10 and packed[:, i].std() > 1e-10:
                feat_corrs.append(float(np.corrcoef(exact[:, i],
                                                    packed[:, i])[0, 1]))
            else:
                feat_corrs.append(float("nan"))
        corrs.append(((n_tr, n_te), feat_corrs))

    n_feat = len(corrs[0][1])
    xs = [n_tr for (n_tr, _), _ in corrs]
    palette = [
        "#4C78A8", "#F58518", "#E45756", "#72B7B2",
        "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6",
        "#9D755D", "#BAB0AC", "#1F77B4", "#FF7F0E",
    ]

    fig = go.Figure()
    for feat_idx in range(n_feat):
        res_idx = feat_idx // 4
        q_idx = feat_idx % 4
        ys = [c[1][feat_idx] for c in corrs]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=f"res {res_idx} · q{q_idx}",
            marker=dict(size=9, color=palette[feat_idx]),
            line=dict(color=palette[feat_idx], width=2),
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#2ca02c",
                  line_width=1.5, opacity=0.6)
    fig.add_annotation(x=max(xs), y=1.0, text="perfect",
                       showarrow=False, yanchor="bottom", xanchor="right",
                       font=dict(size=10, color="#2ca02c"))
    fig.update_xaxes(title_text="Training set size n_train (log)", type="log")
    fig.update_yaxes(title_text="Correlation(exact, packed)",
                     range=[-0.05, 1.05])
    fig.update_layout(
        title="Per-feature fidelity vs dataset size (packed, Z observable, 12 features)",
        height=560, width=1000,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="v", yanchor="middle", y=0.5,
                    xanchor="left", x=1.02,
                    title=dict(text="Feature")),
    )
    save_fig(fig, FIGURES_DIR / "per_feature_corr.html")


def print_summary():
    """Concise text summary of every (sim, source) fidelity pair."""
    pairs = [
        ("SV1 shot-only (5x3 XYZ+ZZ n100+50)",
         "lightning/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_exact.npz",
         "sv1/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz"),
        ("Singleton 4q Rigetti (3x3 Z+ZZ n100+50)",
         "lightning/seed_42/reservoir_3x3_Z+ZZ_n100+50_exact.npz",
         "rigetti/seed_42/reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz"),
        ("Packed 80q Rigetti (3x3 Z n100+50)",
         "lightning/seed_42/reservoir_3x3_Z_n100+50_exact.npz",
         "rigetti/seed_42/reservoir_3x3_Z_n100+50_s1000_packed.npz"),
        ("Packed 80q Rigetti (3x3 Z n500+250)",
         "lightning/seed_42/reservoir_3x3_Z_n500+250_exact.npz",
         "rigetti/seed_42/reservoir_3x3_Z_n500+250_s1000_packed.npz"),
        ("Packed 80q Rigetti (3x3 Z n1000+500)",
         "lightning/seed_42/reservoir_3x3_Z_n1000+500_exact.npz",
         "rigetti/seed_42/reservoir_3x3_Z_n1000+500_s1000_packed.npz"),
    ]
    print(f"\n{'=' * 82}")
    print("FEATURE FIDELITY SUMMARY")
    print(f"{'=' * 82}")
    print(f"  {'Source':<48s} {'corr':>7s} {'mean|err|':>10s} {'max|err|':>10s}")
    for label, sim_path, src_path in pairs:
        ex = load_pooled_aug(FEATURES_DIR / sim_path)
        src = load_pooled_aug(FEATURES_DIR / src_path)
        err = np.abs(ex - src).flatten()
        corr = float(np.corrcoef(ex.flatten(), src.flatten())[0, 1])
        print(f"  {label:<48s} {corr:>7.4f} {err.mean():>10.4f} {err.max():>10.4f}")
    print(f"{'=' * 82}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print_summary()
    plot_singleton_vs_sim_scatter()
    plot_packed_vs_sim_scatter()
    plot_three_way_overlay()
    plot_error_hist()
    plot_per_feature_corr()


if __name__ == "__main__":
    main()
