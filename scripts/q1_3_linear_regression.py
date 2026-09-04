import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.helpers import (
    load_data, save_fig, standardize, gradient_descent, predict, evaluate
)

df, X, y = load_data()
sel = np.load("data/selected_features.npz", allow_pickle=True)
strong = sel["strong"].item()
weak = sel["weak"].item()

# Make a copy with strong predictor index preserved for plotting
y_np = y.values.reshape(-1, 1)

# ── Q1.3.1 Weak predictor with gradient descent + standardization ───────────
print("=" * 60)
print("Q1.3.1 - Simple Linear Regression (Weak Predictor)")
print("=" * 60)

X_weak = X[[weak]].values
X_weak_sc, scaler_weak = standardize(X_weak)
w_w, b_w, hist_w = gradient_descent(X_weak_sc, y_np, lr=0.01, n_iter=2000, return_history=True)
print(f"Weak predictor: {weak}")
print(f"  Coefficient (w): {w_w[0]:.4f}")
print(f"  Intercept (b):   {b_w:.4f}")
print(f"  Final MSE:       {hist_w['cost'][-1]:.4f}")
print("Note: Feature standardized before gradient descent")

# ── Q1.3.2 Strong predictor with gradient descent ───────────────────────────
print("\n" + "=" * 60)
print("Q1.3.2 - Simple Linear Regression (Strong Predictor)")
print("=" * 60)

X_strong = X[[strong]].values
X_strong_sc, scaler_strong = standardize(X_strong)
w_s, b_s, hist_s = gradient_descent(X_strong_sc, y_np, lr=0.01, n_iter=2000, return_history=True)
print(f"Strong predictor: {strong}")
print(f"  Coefficient (w): {w_s[0]:.4f}")
print(f"  Intercept (b):   {b_s:.4f}")
print(f"  Final MSE:       {hist_s['cost'][-1]:.4f}")

# ── Q1.3.3 Compare coefficients ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Q1.3.3 - Model Comparison")
print("=" * 60)
print("              |  Coeff (w)  |  Intercept (b) |  Final MSE  ")
print(f" Weak ({weak:<30s})| {w_w[0]:>11.4f} | {b_w:>14.4f} | {hist_w['cost'][-1]:>11.4f}")
print(f" Strong ({strong:<29s})| {w_s[0]:>11.4f} | {b_s:>14.4f} | {hist_s['cost'][-1]:>11.4f}")
print("(Coefficients are in standardized feature space)")

# ── Q1.3.4 Regression line vs data ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Q1.3.4 - Regression Lines vs Data")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Weak predictor
x_range_w = np.linspace(X_weak_sc.min(), X_weak_sc.max(), 100).reshape(-1, 1)
y_line_w = predict(x_range_w, w_w, b_w)
axes[0].scatter(X_weak_sc, y_np, s=4, alpha=0.3, color="lightcoral")
axes[0].plot(x_range_w, y_line_w, color="red", lw=2, label="Regression line")
axes[0].set_title(f"Weak Predictor: {weak}")
axes[0].set_xlabel("Standardized feature value")
axes[0].set_ylabel("Critical Temperature (K)")
axes[0].legend()

# Strong predictor
x_range_s = np.linspace(X_strong_sc.min(), X_strong_sc.max(), 100).reshape(-1, 1)
y_line_s = predict(x_range_s, w_s, b_s)
axes[1].scatter(X_strong_sc, y_np, s=4, alpha=0.3, color="steelblue")
axes[1].plot(x_range_s, y_line_s, color="darkblue", lw=2, label="Regression line")
axes[1].set_title(f"Strong Predictor: {strong}")
axes[1].set_xlabel("Standardized feature value")
axes[1].set_ylabel("Critical Temperature (K)")
axes[1].legend()

plt.tight_layout()
save_fig("q1_3_regression_lines.png")
print("Figure saved: q1_3_regression_lines.png")
