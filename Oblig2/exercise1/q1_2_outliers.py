"""Ex1 Q1.2 - Outliers in the numeric features.

Flags outliers per feature with the IQR rule and with z-scores, prints the
counts, and saves a figure. The numeric columns have no missing values, so
the raw values can be used directly.
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

from src.data import NUMERIC_COLUMNS, load_raw

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train, _ = load_raw()


def iqr_bounds(x, k=1.5):
    """Low and high fence of the 1.5*IQR rule."""
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


zs = {}
rows = []
for col in NUMERIC_COLUMNS:
    x = train[col]
    lo, hi = iqr_bounds(x)
    z = (x - x.mean()) / x.std()
    zs[col] = z
    rows.append(
        {
            "feature": col,
            "iqr_low": round(lo, 1),
            "iqr_high": round(hi, 1),
            "n_iqr": int(((x < lo) | (x > hi)).sum()),
            "pct_iqr": round(((x < lo) | (x > hi)).mean() * 100, 2),
            "mean": round(x.mean(), 1),
            "std": round(x.std(), 1),
            "n_z": int((z.abs() > 3).sum()),
            "pct_z": round((z.abs() > 3).mean() * 100, 2),
        }
    )

res = pd.DataFrame(rows).set_index("feature")
res.to_csv(OUT_DIR / "q1_2_outlier_counts.csv")
print(res.to_string())

# 1st/99th percentiles as a reference in case we decide to cap the tails
# later (only really matters for the SVM, trees don't care about the scale).
percentiles = pd.DataFrame(
    {col: [train[col].quantile(0.01), train[col].quantile(0.99)] for col in NUMERIC_COLUMNS},
    index=["p01", "p99"],
).T
print("\n1st / 99th percentiles:")
print(percentiles.round(1).to_string())

fig, axes = plt.subplots(2, len(NUMERIC_COLUMNS), figsize=(14, 8))

# Top row: boxplots, the whiskers/fliers follow the 1.5*IQR rule.
for ax, col in zip(axes[0], NUMERIC_COLUMNS):
    ax.boxplot(train[col])
    ax.set_title(col)

# Bottom row: z-score distribution with the +/-3 threshold marked.
for ax, col in zip(axes[1], NUMERIC_COLUMNS):
    ax.hist(np.clip(zs[col], -6, 6), bins=40, color="#4c72b0")
    ax.axvline(-3, color="red", lw=1)
    ax.axvline(3, color="red", lw=1)
    ax.set_xlim(-6, 6)

fig.tight_layout()
fig.savefig(OUT_DIR / "q1_2_outliers.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")