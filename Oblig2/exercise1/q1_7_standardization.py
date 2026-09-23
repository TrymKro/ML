"""Ex1 Q1.7 - Numeric features with and without standardization.

Plots the shape of each numeric feature before and after z-score
standardization (zero mean, unit variance) and saves the mean/std of both
versions. The tree pipeline leaves the numerics untouched; the SVM pipeline
standardizes them after winsorizing (Q1.2).
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import NUMERIC_COLUMNS
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, _, _, _, _, _, _, _, _, _ = make_split()

raw = X_train[NUMERIC_COLUMNS]
std = (raw - raw.mean()) / raw.std()

stats = pd.DataFrame(
    {
        "feature": NUMERIC_COLUMNS,
        "raw_mean": raw.mean().round(2).values,
        "raw_std": raw.std().round(2).values,
        "std_mean": std.mean().round(6).values,
        "std_std": std.std().round(6).values,
    }
)
stats.to_csv(OUT_DIR / "q1_7_standardization.csv", index=False)
print(stats.to_string(index=False))

fig, axes = plt.subplots(2, len(NUMERIC_COLUMNS), figsize=(16, 6))
for j, col in enumerate(NUMERIC_COLUMNS):
    x = raw[col]
    axes[0, j].hist(x, bins=40, color="#4c72b0")
    axes[0, j].set_title(col.replace("capital-", "cap-\n"))
    axes[1, j].hist(std[col], bins=40, color="#55a868")
axes[0, 0].set_ylabel("raw")
axes[1, 0].set_ylabel("standardized")
fig.suptitle("numeric feature distributions, raw vs standardized")
fig.tight_layout()
fig.savefig(OUT_DIR / "q1_7_standardization.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")