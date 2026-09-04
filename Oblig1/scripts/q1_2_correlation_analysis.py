import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import load_data, save_fig, TARGET

df, X, y = load_data()

# ── Q1.2.1 Correlation Heatmap (top 15 with target) ────────────────────────
print("=" * 60)
print("Q1.2.1 - Correlation Heatmap (Top 15 features with target)")
print("=" * 60)

corr_with_target = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
top15 = corr_with_target.head(15).index.tolist()
print("\nTop 15 features correlated with critical_temp:")
for i, feat in enumerate(top15, 1):
    print(f"  {i:2d}. {feat:<35s} corr={df.corr()[TARGET][feat]:.4f}")

corr_top15 = df[top15 + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_top15, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap: Top 15 Features + Target")
plt.tight_layout()
save_fig("q1_2_correlation_heatmap_top15.png")
print("Figure saved: q1_2_correlation_heatmap_top15.png")

# ── Q1.2.2 Strongest correlations with target ──────────────────────────────
print("\n" + "=" * 60)
print("Q1.2.2 - Strongest Correlations with critical_temp")
print("=" * 60)

corr_full = df.corr()[TARGET].drop(TARGET)
strongest_pos = corr_full.idxmax()
strongest_neg = corr_full.idxmin()
print(f"Strongest POSITIVE: {strongest_pos}  (r = {corr_full[strongest_pos]:.4f})")
print(f"Strongest NEGATIVE: {strongest_neg}  (r = {corr_full[strongest_neg]:.4f})")

# ── Q1.2.3 Highly correlated feature pairs (|r| > 0.9) ─────────────────────
print("\n" + "=" * 60)
print("Q1.2.3 - Feature Pairs with |correlation| > 0.9")
print("=" * 60)

corr_matrix = X.corr()
pairs = []
features = corr_matrix.columns.tolist()
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        r = abs(corr_matrix.iloc[i, j])
        if r > 0.9:
            pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["Feature 1", "Feature 2", "Correlation"])
pairs_df = pairs_df.sort_values("Correlation", key=abs, ascending=False)
print(f"\nFound {len(pairs_df)} pairs with |r| > 0.9:\n")
print(pairs_df.head(15).to_string(index=False))

# ── Q1.2.4 Feature selection for regression ────────────────────────────────
print("\n" + "=" * 60)
print("Q1.2.4 - Selected Features for Linear Regression")
print("=" * 60)

strong_predictor = strongest_pos
weak_predictor = corr_with_target.index[-1]
print(f"Strong predictor: {strong_predictor} (r={corr_full[strong_predictor]:.4f})")
print(f"Weak predictor:   {weak_predictor} (r={corr_full[weak_predictor]:.4f})")

# Save selections for downstream scripts
np.savez("data/selected_features.npz",
         strong=strong_predictor, weak=weak_predictor)
print("\nFeature selections saved to data/selected_features.npz")
