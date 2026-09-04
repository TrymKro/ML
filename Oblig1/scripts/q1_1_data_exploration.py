import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import load_data, save_fig, TARGET

df, X, y = load_data()

# ── Q1.1.1 Summary Statistics ──────────────────────────────────────────────
print("=" * 60)
print("Q1.1.1 - Summary Statistics")
print("=" * 60)

stats = df.describe().T
stats["range"] = stats["max"] - stats["min"]
stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)

print("\nFull statistics saved to figures/stats_full.csv")
stats.to_csv("figures/stats_full.csv")

top_std = stats.sort_values("std", ascending=False).head(10)
print("\nTop 10 features by standard deviation:")
print(top_std[["mean", "std", "min", "max", "cv"]].to_string())

top_cv = stats.sort_values("cv", ascending=False).head(10)
print("\nTop 10 features by coefficient of variation (CV):")
print(top_cv[["mean", "std", "min", "max", "cv"]].to_string())

# ── Q1.1.2 Critical Temperature Distribution ───────────────────────────────
print("\n" + "=" * 60)
print("Q1.1.2 - Critical Temperature Distribution")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(y, bins=60, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].set_title("Critical Temperature - Raw")
axes[0].set_xlabel("Critical Temperature (K)")
axes[0].set_ylabel("Frequency")

y_log = np.log1p(y)
axes[1].hist(y_log, bins=60, edgecolor="black", alpha=0.7, color="coral")
axes[1].set_title("Critical Temperature - Log Transformed")
axes[1].set_xlabel("log(1 + Critical Temperature)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
save_fig("q1_1_critical_temp_distribution.png")
print("Figure saved: q1_1_critical_temp_distribution.png")

print(f"\nRaw skewness:       {y.skew():.4f}")
print(f"Log skewness:       {pd.Series(y_log).skew():.4f}")
print(f"Raw mean:           {y.mean():.4f}")
print(f"Log mean:           {pd.Series(y_log).mean():.4f}")
