"""
Section 4: Classical Baseline

Polynomial feature expansion + linear regression.
Establishes the bar that quantum features must beat.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from exploration.e01_data_generation import generate_regime_data, N_TRAIN, N_TOTAL
from sklearn.model_selection import train_test_split

SEED = 42


def evaluate_model(y_true, y_pred, label=""):
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0]
    return {"model": label, "MSE": round(mse, 6), "MAE": round(mae, 6), "Corr": round(corr, 4)}


if __name__ == "__main__":
    X, Y, regimes = generate_regime_data(N_TOTAL)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=N_TRAIN, shuffle=False, random_state=SEED
    )

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []

    # Degree-1 (raw features)
    lr_raw = LinearRegression().fit(X_train_s, y_train)
    pred_raw = lr_raw.predict(X_test_s)
    results.append(evaluate_model(y_test, pred_raw, "LR (raw features)"))

    # Degree-2 (polynomial features)
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_s)
    X_test_poly = poly.transform(X_test_s)

    lr_poly = LinearRegression().fit(X_train_poly, y_train)
    pred_poly = lr_poly.predict(X_test_poly)
    results.append(evaluate_model(y_test, pred_poly, "LR (poly deg=2)"))

    # Ridge with poly (regularized)
    ridge_poly = Ridge(alpha=1.0).fit(X_train_poly, y_train)
    pred_ridge = ridge_poly.predict(X_test_poly)
    results.append(evaluate_model(y_test, pred_ridge, "Ridge (poly deg=2)"))

    # Lasso with poly
    lasso_poly = Lasso(alpha=0.01).fit(X_train_poly, y_train)
    pred_lasso = lasso_poly.predict(X_test_poly)
    results.append(evaluate_model(y_test, pred_lasso, "Lasso (poly deg=2)"))

    baseline_df = pd.DataFrame(results)
    print("=== Classical Baselines (Out-of-Sample) ===")
    print(baseline_df.to_string(index=False))
