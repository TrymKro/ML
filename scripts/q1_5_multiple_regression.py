import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils.helpers import (
    load_data, save_fig, standardize, gradient_descent, predict, evaluate
)
import matplotlib.pyplot as plt

df, X, y = load_data()
sel = np.load("data/selected_features.npz", allow_pickle=True)
strong = sel["strong"].item()
weak = sel["weak"].item()

y_np = y.values.reshape(-1, 1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {"simple_strong": [], "simple_weak": [], "multiple": []}

print("=" * 70)
print("Q1.5 - Multiple Linear Regression (All 81 Features)")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]

    # Multiple regression (all features)
    X_train_sc, X_test_sc, scaler = standardize(X_train, X_test)
    w_m, b_m, hist_m = gradient_descent(X_train_sc, y_train, lr=0.01, n_iter=3000, return_history=True)
    y_pred_m = predict(X_test_sc, w_m, b_m)
    m_m = evaluate(y_test, y_pred_m)
    m_m["fold"] = fold
    results["multiple"].append(m_m)
    print(f"Fold {fold} (multiple): MSE={m_m['MSE']:.2f}  RMSE={m_m['RMSE']:.2f}  R2={m_m['R2']:.4f}")

# Recompute simple models for comparison (reuse from Q1.4 via a rerun here)
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    y_train, y_test = y_np[train_idx], y_np[test_idx]
    for name, feature in [("simple_strong", strong), ("simple_weak", weak)]:
        X_feat = X[[feature]].values
        X_train_s, X_test_s = X_feat[train_idx], X_feat[test_idx]
        X_train_sc, X_test_sc, _ = standardize(X_train_s, X_test_s)
        w, b = gradient_descent(X_train_sc, y_train, lr=0.01, n_iter=2000)
        y_pred = predict(X_test_sc, w, b)
        m = evaluate(y_test, y_pred)
        m["fold"] = fold
        results[name].append(m)

# ── Q1.5.1 / Q1.5.2 Comparison ──────────────────────────────────────────────
print("\n" + "-" * 70)
print("Q1.5.1/Q1.5.2 - Aggregate Results across 5 Folds")
print("-" * 70)

summary = {}
for name, res in results.items():
    res_df = pd.DataFrame(res)
    summary[name] = res_df[["MSE", "RMSE", "R2"]].mean()
    print(f"\n{name} (mean over folds):")
    for metric in ["MSE", "RMSE", "R2"]:
        print(f"  {metric}: {summary[name][metric]:.4f}")

# ── Q1.5.3 Comparison plots ─────────────────────────────────────────────────
# Plot (iii) Predicted vs Actual for simple (strength) and multiple
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Use fold 1 (representative) for predicted vs actual
train_idx, test_idx = next(kf.split(X))
y_test = y_np[test_idx]

# Multiple
X_train_sc, X_test_sc, _ = standardize(X.values[train_idx], X.values[test_idx])
w_m, b_m, _ = gradient_descent(X_train_sc, y_np[train_idx], lr=0.01, n_iter=3000, return_history=True)
y_pred_m = predict(X_test_sc, w_m, b_m)
axes[0].scatter(y_test, y_pred_m, s=8, alpha=0.4, color="steelblue")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect prediction")
axes[0].set_title("Multiple Regression: Predicted vs Actual")
axes[0].set_xlabel("Actual Critical Temp (K)")
axes[0].set_ylabel("Predicted Critical Temp (K)")
axes[0].legend()

# Simple (strong)
X_s = X[[strong]].values
X_train_sc, X_test_sc, _ = standardize(X_s[train_idx], X_s[test_idx])
w_s, b_s, _ = gradient_descent(X_train_sc, y_np[train_idx], lr=0.01, n_iter=2000, return_history=True)
y_pred_s = predict(X_test_sc, w_s, b_s)
axes[1].scatter(y_test, y_pred_s, s=8, alpha=0.4, color="coral")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect prediction")
axes[1].set_title("Simple Regression (Strong): Predicted vs Actual")
axes[1].set_xlabel("Actual Critical Temp (K)")
axes[1].set_ylabel("Predicted Critical Temp (K)")
axes[1].legend()

plt.tight_layout()
save_fig("q1_5_predicted_vs_actual.png")
print("Figure saved: q1_5_predicted_vs_actual.png")

# Plot (i) Cost vs Iteration for multiple regression
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hist_m["cost"], color="steelblue")
ax.set_title("Multiple Regression: Cost vs Iteration")
ax.set_xlabel("Iteration")
ax.set_ylabel("MSE Cost")
ax.grid(alpha=0.3)
save_fig("q1_5_cost_vs_iteration.png")
print("Figure saved: q1_5_cost_vs_iteration.png")

# Plot (iv) Residuals plot for multiple regression
fig, ax = plt.subplots(figsize=(8, 5))
residuals = y_test.flatten() - y_pred_m.flatten()
ax.scatter(y_pred_m, residuals, s=8, alpha=0.4, color="purple")
ax.axhline(0, color="red", ls="--", lw=2)
ax.set_title("Multiple Regression: Residuals Plot")
ax.set_xlabel("Predicted Critical Temp (K)")
ax.set_ylabel("Residuals")
ax.grid(alpha=0.3)
save_fig("q1_5_residuals.png")
print("Figure saved: q1_5_residuals.png")
