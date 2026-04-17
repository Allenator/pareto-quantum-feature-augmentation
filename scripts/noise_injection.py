"""Find the classical Gaussian σ that matches each hardware source's Ridge MSE.

Idea
----
For each Rigetti run, start from the exact-simulator features, inject
i.i.d. Gaussian noise of increasing σ into the quantum columns only, and
re-fit Ridge at each σ. The σ that reproduces the observed hardware MSE is
the "effective classical noise equivalent" of the quantum hardware for that
run — a single-number summary of how bad the hardware is.

This complements Analysis 1 (which decomposed hardware noise into damping λ
and a gate term σ_g). The σ from this analysis mixes both effects:
approximately σ*² ≈ σ_g² + shot² + (1 − λ)²·⟨E²⟩.

Outputs
-------
  results/synthetic_hw/noise_injection.csv  — σ grid, matched σ*, MSEs
  plots/synthetic_hw/noise_injection.{html,png}
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
N_TRIALS = 16       # noise realisations averaged per σ
SIGMA_GRID = np.linspace(0.0, 0.80, 33)

# (label, n_train, n_test, sim_rel, hw_rel)
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

COLORS = {
    "singleton_z+zz_n100+50": "#B279A2",
    "packed_z_n100+50":       "#E45756",
    "packed_z_n500+250":      "#F58518",
    "packed_z_n1000+500":     "#9D755D",
}


def ridge_mse(X_train, y_train, X_test, y_test) -> float:
    pred = RidgeModel().fit_predict(X_train, y_train, X_test)
    return float(np.mean((pred.y_pred - y_test) ** 2))


def sweep_sigma(sim_train, sim_test, y_train, y_test,
                sigma_grid, n_trials=N_TRIALS) -> np.ndarray:
    """Return mean Ridge test MSE across n_trials noise realisations, per σ."""
    raw_train = sim_train[:, :N_ORIG]
    raw_test = sim_test[:, :N_ORIG]
    q_train = sim_train[:, N_ORIG:]
    q_test = sim_test[:, N_ORIG:]
    mses = np.zeros_like(sigma_grid)
    for i, sigma in enumerate(sigma_grid):
        trial_mses = []
        for trial in range(n_trials):
            rng = np.random.default_rng(SEED + trial)
            noisy_q_train = np.clip(
                q_train + sigma * rng.standard_normal(q_train.shape), -1.0, 1.0)
            noisy_q_test = np.clip(
                q_test + sigma * rng.standard_normal(q_test.shape), -1.0, 1.0)
            X_tr = np.hstack([raw_train, noisy_q_train])
            X_te = np.hstack([raw_test, noisy_q_test])
            trial_mses.append(ridge_mse(X_tr, y_train, X_te, y_test))
        mses[i] = float(np.mean(trial_mses))
    return mses


def find_matching_sigma(sigma_grid: np.ndarray, mse_curve: np.ndarray,
                         target_mse: float) -> float | None:
    """Linearly interpolate σ* where mse_curve crosses target_mse.

    Returns None if target is below the curve's floor (over-noisy hardware)
    or above the curve's ceiling (under-noisy hardware).
    """
    # mse_curve is generally monotonically increasing in σ, but not guaranteed.
    # Take the first crossing from below.
    for i in range(len(sigma_grid) - 1):
        y0, y1 = mse_curve[i], mse_curve[i + 1]
        if (y0 <= target_mse <= y1) or (y1 <= target_mse <= y0):
            x0, x1 = sigma_grid[i], sigma_grid[i + 1]
            if y1 == y0:
                return float(x0)
            frac = (target_mse - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    curve_rows = []
    fig = go.Figure()

    for label, n_tr, n_te, sim_rel, hw_rel in SOURCES:
        sim = np.load(FEATURES_DIR / sim_rel)
        hw = np.load(FEATURES_DIR / hw_rel)

        dgp = DGPConfig(n_train=n_tr, n_test=n_te, seed=SEED)
        data = get_or_generate(dgp, "data/synthetic")
        y_train, y_test = data[2], data[3]

        # Hardware baseline MSE
        hw_mse = ridge_mse(hw["train"], y_train, hw["test"], y_test)
        exact_mse = ridge_mse(sim["train"], y_train, sim["test"], y_test)

        # MSE curve as σ varies
        mse_curve = sweep_sigma(sim["train"], sim["test"], y_train, y_test,
                                SIGMA_GRID)

        # Find σ* matching hw_mse
        sigma_star = find_matching_sigma(SIGMA_GRID, mse_curve, hw_mse)

        summary_rows.append({
            "source": label,
            "n_train": n_tr,
            "n_test": n_te,
            "exact_mse": round(exact_mse, 4),
            "hw_mse": round(hw_mse, 4),
            "sigma_star": round(sigma_star, 4) if sigma_star is not None else "",
            "notes": ("" if sigma_star is not None
                      else "hw MSE outside σ grid range"),
        })
        for s, m in zip(SIGMA_GRID, mse_curve):
            curve_rows.append({
                "source": label,
                "sigma": round(float(s), 4),
                "ridge_mse": round(float(m), 4),
            })

        # Add MSE curve trace
        fig.add_trace(go.Scatter(
            x=SIGMA_GRID, y=mse_curve, mode="lines+markers",
            name=(f"{label}  σ*={sigma_star:.3f}" if sigma_star is not None
                  else f"{label}  (no σ* in grid)"),
            marker=dict(size=6, color=COLORS[label]),
            line=dict(color=COLORS[label], width=2),
        ))
        # Hardware-MSE horizontal reference line
        fig.add_hline(y=hw_mse, line_dash="dot", line_color=COLORS[label],
                      line_width=1, opacity=0.5)
        # Annotate σ* intersection
        if sigma_star is not None:
            fig.add_trace(go.Scatter(
                x=[sigma_star], y=[hw_mse], mode="markers",
                marker=dict(size=12, color=COLORS[label], symbol="x-thin",
                            line=dict(color="white", width=1)),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        title=("Matched classical noise injection — Ridge MSE of (exact + "
               "𝒩(0, σ)) vs σ.  σ* (×) = σ that reproduces Rigetti MSE."),
        xaxis_title="σ of injected Gaussian noise on quantum features only",
        yaxis_title="Ridge test MSE (lower = less noisy)",
        height=560, width=1200,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.01),
    )
    path = FIGURES_DIR / "noise_injection.html"
    fig.write_html(str(path), include_plotlyjs=True)
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path} + .png")
    except Exception as e:
        print(f"Saved: {path}  (PNG export failed: {e})")

    # CSVs
    summary_path = RESULTS_DIR / "noise_injection.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")

    curve_path = RESULTS_DIR / "noise_injection_curve.csv"
    with open(curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"Saved: {curve_path}")

    # Report
    print(f"\n{'=' * 90}")
    print("MATCHED CLASSICAL NOISE INJECTION")
    print(f"{'=' * 90}")
    print(f"  {'Source':<26s} {'exact MSE':>10s} {'hw MSE':>10s} "
          f"{'σ*':>8s}  {'notes':s}")
    for r in summary_rows:
        ss = r["sigma_star"]
        ss_str = f"{ss:.3f}" if ss != "" else "n/a"
        print(f"  {r['source']:<26s} {r['exact_mse']:>10.4f} "
              f"{r['hw_mse']:>10.4f} {ss_str:>8s}  {r['notes']}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
