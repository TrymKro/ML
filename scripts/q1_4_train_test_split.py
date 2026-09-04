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

results = {"strong": [], "weak": []}

print("=" * 70)
print("Q1.4 - 5-Fold Train-Test Evaluation (Simple Linear Regression)")
print("=" * 70)

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    for name, feature in [("strong", strong), ("weak", weak)]:
        X_feat = X[[feature]].values
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        X_train_sc, X_test_sc, scaler = standardize(X_train, X_test)

        w, b = gradient_descent(X_train_sc, y_train, lr=0.01, n_iter=2000)
        y_pred = predict(X_test_sc, w, b)

        m = evaluate(y_test, y_pred)
        m["fold"] = fold
        results[name].append(m)
        print(f"Fold {fold} ({name:6s}): MSE={m['MSE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

# ── Q1.4.1 / Q1.4.2 Summary ─────────────────────────────────────────────────
print("\n" + "-" * 70)
strong_df = pd.DataFrame(results["strong"])
weak_df = pd.DataFrame(results["weak"])

print("\nQ1.4.1 - Strong Predictor Summary per Fold:")
print(strong_df.to_string(index=False))

print("\nQ1.4.2 - Weak Predictor Summary per Fold:")
print(weak_df.to_string(index=False))

# ── Q1.4.4 Mean and variance across folds ───────────────────────────────────
print("\n" + "-" * 70)
print("Q1.4.4 - Mean and Variance of Metrics across 5 Folds")

for name, res_df in [("Strong predictor", strong_df), ("Weak predictor", weak_df)]:
    print(f"\n  {name}:")
    for metric in ["MSE", "RMSE", "R2"]:
        mean_v = res_df[metric].mean()
        var_v = res_df[metric].var()
        print(f"    {metric}: mean={mean_v:.4f}, variance={var_v:.4f}")

# ── Plot per-fold comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fold_labels = strong_df["fold"].values

for ax, metric in zip(axes, ["MSE", "RMSE", "R2"]):
    ax.plot(fold_labels, strong_df[metric], "o-", label="Strong", color="steelblue")
    ax.plot(fold_labels, weak_df[metric], "s--", label="Weak", color="coral")
    ax.set_title(f"{metric} per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
save_fig("q1_4_fold_comparison.png")
print("\nFigure saved: q1_4_fold_comparison.png")
