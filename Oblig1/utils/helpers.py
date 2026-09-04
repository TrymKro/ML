import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIG_DIR = os.path.join(ROOT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "critical_temp"


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "superconductivity.csv"))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return df, X, y


def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def standardize(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_sc = scaler.transform(X_test)
        return X_train_sc, X_test_sc, scaler
    return X_train_sc, scaler


def gradient_descent(X, y, lr=0.01, n_iter=1000, return_history=False):
    n, d = X.shape
    y = y.reshape(n) if y.ndim == 2 else y
    w = np.zeros(d)
    b = 0.0
    history = {"cost": [], "w": []}

    for i in range(n_iter):
        y_pred = X @ w + b
        error = y_pred - y
        cost = np.mean(error ** 2)
        history["cost"].append(cost)
        history["w"].append(w.copy())

        grad_w = (2 / n) * (X.T @ error)
        grad_b = (2 / n) * np.sum(error)
        w -= lr * grad_w
        b -= lr * grad_b

    if return_history:
        return w, b, history
    return w, b


def predict(X, w, b):
    return X @ w + b


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"MSE": mse, "RMSE": rmse, "R2": r2}


def print_metrics(metrics, label=""):
    print(f"{label}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
