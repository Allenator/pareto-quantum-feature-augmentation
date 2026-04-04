"""Shared imports and constants used across all exploration scripts."""

# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classical ML
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Quantum ML
import pennylane as qml
from pennylane import numpy as pnp

# Amazon Braket
from braket.circuits import Circuit, Observable, ResultType
from braket.devices import LocalSimulator

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Quantum config
N_QUBITS = 4
N_FEATURES = 4  # X1, X2, X3, X4

# Local state-vector simulator (exact, no shot noise)
SV_DEVICE = LocalSimulator("default")


def evaluate_model(y_true, y_pred, label=""):
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0]
    return {"model": label, "MSE": round(mse, 6), "MAE": round(mae, 6), "Corr": round(corr, 4)}
