"""Calibrate a simple two-parameter noise model against SV1 and Rigetti runs.

Model
-----
For a single Pauli expectation value produced by an exact circuit, we posit

    R = λ · E + δ_shot + δ_gate

where
    E          : noise-free expectation from the exact simulator
    λ ∈ (0, 1] : multiplicative damping (depolarising-like attenuation toward 0)
    δ_shot     : shot-sampling error, centred, per-point variance (1 − (λE)²)/n_shots
    δ_gate     : additional Gaussian error from gate/readout/crosstalk, variance σ_g²

Fit
---
For pooled (E, R) pairs, we use
    λ   = (E · R) / (E · E)                        (OLS through origin)
    Var(R − λE) = ⟨σ_shot²(λE)⟩ + σ_g²              (decompose residual variance)
to recover λ and σ_g with n_shots = 1000 treated as known (verified on SV1).

Diagnostic
----------
Apply the fitted model to exact-sim features, sample synthetic "noisy" output,
and overlay its error histogram against the real hardware histogram.
Also report the predicted pooled correlation and mean |error|, which should
match the hardware values if the simple model captures the dominant noise.

Usage
-----
    uv run python scripts/fit_noise_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import numpy as np
import plotly.graph_objects as go

FEATURES_DIR = Path("features/synthetic_hw")
RESULTS_DIR = Path("results/synthetic_hw")
FIGURES_DIR = Path("plots/synthetic_hw")
N_ORIG = 4
N_SHOTS = 1000

COLORS = {
    "hardware": "#E45756",   # coral — real hardware
    "model":    "#4C78A8",   # blue — fitted noise model
    "shot":     "#54A24B",   # green — shot-noise only
}


def load_aug(path: Path) -> np.ndarray:
    """train+test augmented-only pooled as (n_samples, n_aug)."""
    d = np.load(path)
    return np.vstack([d["train"][:, N_ORIG:], d["test"][:, N_ORIG:]])


def fit_noise_params(exact: np.ndarray, hardware: np.ndarray,
                     n_shots: int = N_SHOTS) -> dict:
    """Fit (λ, σ_g) from pooled (exact, hardware) pairs.

    Returns a dict with the parameters and residual diagnostics.
    """
    e = exact.flatten()
    r = hardware.flatten()
    lam = float(e @ r) / float(e @ e)

    residual = r - lam * e
    resid_var = float(np.var(residual))

    # Predicted shot variance averaged over pooled points.
    shot_var_per_point = (1.0 - np.clip((lam * e) ** 2, 0, 1)) / n_shots
    mean_shot_var = float(shot_var_per_point.mean())

    gate_var = max(0.0, resid_var - mean_shot_var)
    sigma_gate = float(np.sqrt(gate_var))
    sigma_shot = float(np.sqrt(mean_shot_var))
    sigma_total = float(np.sqrt(resid_var))

    # Pooled-measurement summary
    corr = float(np.corrcoef(e, r)[0, 1])
    mean_err = float(np.abs(r - e).mean())

    return {
        "n_points": len(e),
        "lambda": lam,
        "sigma_gate": sigma_gate,
        "sigma_shot": sigma_shot,
        "sigma_total": sigma_total,
        "corr_observed": corr,
        "mean_abs_err_observed": mean_err,
    }


def apply_noise_model(exact: np.ndarray, lam: float, sigma_gate: float,
                      n_shots: int = N_SHOTS, seed: int = 0) -> np.ndarray:
    """Generate one synthetic 'noisy' realisation of exact features."""
    rng = np.random.default_rng(seed)
    damped = lam * exact
    shot_sigma = np.sqrt(np.clip(1.0 - damped ** 2, 0, 1) / n_shots)
    total_sigma = np.sqrt(shot_sigma ** 2 + sigma_gate ** 2)
    noise = rng.standard_normal(exact.shape) * total_sigma
    noisy = damped + noise
    # Clip to the Pauli expectation range
    return np.clip(noisy, -1.0, 1.0)


def validate_fit(exact: np.ndarray, hardware: np.ndarray,
                 params: dict, n_trials: int = 32) -> dict:
    """Draw multiple noise realisations and compare to hardware summary."""
    lam, sg = params["lambda"], params["sigma_gate"]
    corr_samples, err_samples = [], []
    for seed in range(n_trials):
        noisy = apply_noise_model(exact, lam, sg, seed=seed)
        corr_samples.append(np.corrcoef(exact.flatten(),
                                        noisy.flatten())[0, 1])
        err_samples.append(np.abs(exact - noisy).mean())
    return {
        "corr_model_mean": float(np.mean(corr_samples)),
        "corr_model_std": float(np.std(corr_samples)),
        "mean_abs_err_model_mean": float(np.mean(err_samples)),
        "mean_abs_err_model_std": float(np.std(err_samples)),
    }


def plot_fit_diagnostics(fits: list, n_shots: int = N_SHOTS):
    """Overlay hardware and model-synthesised error histograms per source."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    bins = dict(start=0.0, end=1.0, size=0.025)

    for entry in fits:
        label = entry["label"]
        exact = entry["exact"]
        hw = entry["hardware"]
        params = entry["params"]

        err_hw = np.abs(exact - hw).flatten()
        # One realisation from the calibrated model
        noisy = apply_noise_model(exact, params["lambda"],
                                  params["sigma_gate"], seed=0)
        err_model = np.abs(exact - noisy).flatten()

        fig.add_trace(go.Histogram(
            x=err_hw, histnorm="probability density",
            name=f"{label} · hardware  (mean={err_hw.mean():.3f})",
            marker=dict(color=COLORS["hardware"],
                        line=dict(color="#7a2e30", width=0.5)),
            opacity=0.55, xbins=bins, legendgroup=label,
        ))
        fig.add_trace(go.Histogram(
            x=err_model, histnorm="probability density",
            name=(f"{label} · model  λ={params['lambda']:.3f}, "
                  f"σ_g={params['sigma_gate']:.3f}  "
                  f"(mean={err_model.mean():.3f})"),
            marker=dict(color=COLORS["model"],
                        line=dict(color="#2a4a70", width=0.5)),
            opacity=0.55, xbins=bins, legendgroup=label,
        ))

    # Shot-noise reference
    fig.add_vline(x=np.sqrt(2 / (np.pi * n_shots)), line_dash="dash",
                  line_color="#444", line_width=1.5, opacity=0.7)
    fig.add_annotation(x=np.sqrt(2 / (np.pi * n_shots)), y=1.0,
                       xref="x", yref="paper",
                       text=f"shot-noise mean |err| ≈ {np.sqrt(2/(np.pi*n_shots)):.3f}",
                       showarrow=False, xanchor="left", yanchor="bottom",
                       xshift=4, font=dict(size=10, color="#444"))

    fig.update_layout(
        title=("Noise-model fit diagnostic — hardware vs (λ·exact + shot + gate) histograms"),
        xaxis_title="|exact − source| per measurement",
        yaxis_title="Probability density",
        barmode="overlay", bargap=0.05,
        height=600, width=1200,
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=0.99,
                    xanchor="right", x=0.99,
                    bordercolor="#ccc", borderwidth=1),
    )
    path = FIGURES_DIR / "noise_model_fit.html"
    fig.write_html(str(path), include_plotlyjs=True)
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=3)
        print(f"Saved: {path} + .png")
    except Exception as e:
        print(f"Saved: {path}  (PNG export failed: {e})")


def write_csv(fits: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "n_shots", "n_points",
        "lambda", "sigma_gate", "sigma_shot", "sigma_total",
        "corr_observed", "mean_abs_err_observed",
        "corr_model_mean", "corr_model_std",
        "mean_abs_err_model_mean", "mean_abs_err_model_std",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in fits:
            row = {"label": entry["label"], "n_shots": N_SHOTS}
            row.update(entry["params"])
            row.update(entry["validation"])
            writer.writerow(row)
    print(f"Saved: {path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Define the three fits.  For packed, pool across all three sizes because
    # hardware noise should be dataset-invariant — pooling that way tests
    # the assumption and gives us a larger sample.
    sources = [
        {
            "label": "SV1 shot-only",
            "pairs": [(
                "lightning/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_exact.npz",
                "sv1/seed_42/reservoir_5x3_XYZ+ZZ_n100+50_s1000_sv1.npz",
            )],
        },
        {
            "label": "Singleton 4q Rigetti",
            "pairs": [(
                "lightning/seed_42/reservoir_3x3_Z+ZZ_n100+50_exact.npz",
                "rigetti/seed_42/reservoir_3x3_Z+ZZ_n100+50_s1000_singleton.npz",
            )],
        },
        {
            "label": "Packed 80q Rigetti (pooled Z)",
            "pairs": [
                ("lightning/seed_42/reservoir_3x3_Z_n100+50_exact.npz",
                 "rigetti/seed_42/reservoir_3x3_Z_n100+50_s1000_packed.npz"),
                ("lightning/seed_42/reservoir_3x3_Z_n500+250_exact.npz",
                 "rigetti/seed_42/reservoir_3x3_Z_n500+250_s1000_packed.npz"),
                ("lightning/seed_42/reservoir_3x3_Z_n1000+500_exact.npz",
                 "rigetti/seed_42/reservoir_3x3_Z_n1000+500_s1000_packed.npz"),
            ],
        },
    ]

    fits = []
    for src in sources:
        ex_parts, hw_parts = [], []
        for ex_path, hw_path in src["pairs"]:
            ex_parts.append(load_aug(FEATURES_DIR / ex_path))
            hw_parts.append(load_aug(FEATURES_DIR / hw_path))
        exact = np.vstack(ex_parts)
        hardware = np.vstack(hw_parts)

        params = fit_noise_params(exact, hardware)
        validation = validate_fit(exact, hardware, params)

        fits.append({
            "label": src["label"],
            "exact": exact,
            "hardware": hardware,
            "params": params,
            "validation": validation,
        })

    # Report
    print(f"\n{'=' * 88}")
    print("NOISE-MODEL CALIBRATION RESULTS")
    print(f"{'=' * 88}")
    print(f"  {'Source':<36s} {'λ':>7s} {'σ_g':>7s} {'σ_shot':>8s} {'σ_tot':>7s} "
          f"{'r_hw':>7s} {'r_mod':>7s}")
    for entry in fits:
        p, v = entry["params"], entry["validation"]
        print(f"  {entry['label']:<36s} "
              f"{p['lambda']:>7.4f} {p['sigma_gate']:>7.4f} "
              f"{p['sigma_shot']:>8.4f} {p['sigma_total']:>7.4f} "
              f"{p['corr_observed']:>7.4f} {v['corr_model_mean']:>7.4f}")
    print(f"{'=' * 88}")
    print(f"  n_shots = {N_SHOTS} (assumed, verified on SV1)")
    print(f"  r_hw   = observed pooled correlation (exact vs hardware)")
    print(f"  r_mod  = model-predicted correlation (mean over 32 resamplings)")
    print(f"{'=' * 88}\n")

    plot_fit_diagnostics(fits)
    write_csv(fits, RESULTS_DIR / "noise_model_fit.csv")


if __name__ == "__main__":
    main()
