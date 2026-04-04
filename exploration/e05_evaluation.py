"""
Section 9: Evaluation

Compare classical and quantum feature augmentation approaches
with consistent out-of-sample metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from exploration.e01_data_generation import generate_regime_data, N_TRAIN, N_TOTAL
from exploration.e03_quantum_feature_maps import (
    extract_quantum_features,
    circuit_angle_basic_entangle,
    circuit_zz_feature_map,
    circuit_iqp_encoding,
    N_QUBITS,
)
from pennylane import numpy as pnp

SEED = 42


def evaluate_model(y_true, y_pred, label=""):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0]
    return {"model": label, "MSE": round(mse, 6), "MAE": round(mae, 6), "Corr": round(corr, 4)}


if __name__ == "__main__":
    # Use smaller subset for quantum circuits (expensive)
    N_EVAL = 2000
    X, Y, _ = generate_regime_data(N_EVAL)
    split = N_EVAL // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # --- Classical baselines ---
    lr = LinearRegression().fit(X_train_s, y_train)
    results.append(evaluate_model(y_test, lr.predict(X_test_s), "LR (raw)"))

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)
    ridge = Ridge(alpha=1.0).fit(X_train_poly, y_train)
    results.append(evaluate_model(y_test, ridge.predict(X_test_poly), "Ridge (poly deg=2)"))

    # --- Quantum: Angle encoding ---
    print("Extracting angle encoding features...")
    n_layers = 2
    weights = pnp.random.uniform(0, 2 * np.pi, (n_layers, N_QUBITS), requires_grad=False)
    q_train_angle = extract_quantum_features(circuit_angle_basic_entangle, X_train_s, weights=weights)
    q_test_angle = extract_quantum_features(circuit_angle_basic_entangle, X_test_s, weights=weights)

    X_train_q = np.hstack([X_train_s, q_train_angle])
    X_test_q = np.hstack([X_test_s, q_test_angle])
    ridge_q = Ridge(alpha=1.0).fit(X_train_q, y_train)
    results.append(evaluate_model(y_test, ridge_q.predict(X_test_q), "Ridge + Angle Encoding"))

    # --- Quantum: ZZ feature map ---
    print("Extracting ZZ feature map features...")
    q_train_zz = extract_quantum_features(circuit_zz_feature_map, X_train_s, reps=2)
    q_test_zz = extract_quantum_features(circuit_zz_feature_map, X_test_s, reps=2)

    X_train_zz = np.hstack([X_train_s, q_train_zz])
    X_test_zz = np.hstack([X_test_s, q_test_zz])
    ridge_zz = Ridge(alpha=1.0).fit(X_train_zz, y_train)
    results.append(evaluate_model(y_test, ridge_zz.predict(X_test_zz), "Ridge + ZZ Map"))

    # --- Summary ---
    df = pd.DataFrame(results)
    print("\n=== Evaluation Summary ===")
    print(df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(["MSE", "MAE", "Corr"]):
        axes[i].barh(df["model"], df[metric])
        axes[i].set_title(metric)
        axes[i].invert_yaxis()
    plt.tight_layout()
    plt.savefig("exploration/evaluation_comparison.png", dpi=150)
    print("Saved plot to exploration/evaluation_comparison.png")
