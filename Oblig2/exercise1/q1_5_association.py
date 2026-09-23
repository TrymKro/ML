"""Ex1 Q1.5 - Association between the features.

Builds a single 14x14 association matrix for the mixed-type data: Pearson |r|
for numeric-numeric pairs, Cramer's V for categorical-categorical pairs, and
the correlation ratio (eta) for numeric-categorical pairs. All values lie in
[0, 1] so one heatmap works for every combination.
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
from scipy.stats import chi2_contingency

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.preprocessing import MIN_COUNT
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, _, _, _, _, _, _, _, _, _ = make_split()

# Same cleaning as the modelling pipeline: 'Unknown' missing markers (Q1.1)
# and the rare-category grouping (Q1.3).
for col in ["workclass", "occupation", "native-country"]:
    X_train[col] = X_train[col].fillna("Unknown")
for col in CATEGORICAL_COLUMNS:
    counts = X_train[col].value_counts()
    keep = counts[counts >= MIN_COUNT].index
    X_train[col] = X_train[col].where(X_train[col].isin(keep), "other")

FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def cramers_v(a, b):
    """Cramer's V between two categorical series."""
    tab = pd.crosstab(a, b)
    chi2, *_ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / n / min(r - 1, k - 1)))


def correlation_ratio(cat, num):
    """Correlation ratio (eta) of the numeric series grouped by cat."""
    groups = num.groupby(cat)
    grand = num.mean()
    ss_between = ((groups.mean() - grand) ** 2 * groups.size()).sum()
    ss_total = ((num - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total))


assoc = pd.DataFrame(np.eye(len(FEATURES)), index=FEATURES, columns=FEATURES)
for i, a in enumerate(FEATURES):
    for j, b in enumerate(FEATURES):
        if i <= j:
            continue
        if a in NUMERIC_COLUMNS and b in NUMERIC_COLUMNS:
            v = abs(X_train[a].corr(X_train[b]))
        elif a in CATEGORICAL_COLUMNS and b in CATEGORICAL_COLUMNS:
            v = cramers_v(X_train[a], X_train[b])
        else:
            num, cat = (a, b) if a in NUMERIC_COLUMNS else (b, a)
            v = correlation_ratio(X_train[cat], X_train[num])
        assoc.loc[a, b] = assoc.loc[b, a] = v

assoc.to_csv(OUT_DIR / "q1_5_association.csv")

# The strongest pairs, for the report.
vals = assoc.where(np.triu(np.ones(assoc.shape, dtype=bool), 1))
pairs = vals.stack().sort_values(ascending=False)
print(pairs.head(12).round(3).to_string())

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(assoc.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{assoc.values[i, j]:.2f}", ha="center", va="center",
                fontsize=6.5)
fig.colorbar(im, ax=ax, shrink=0.8, label="association (0-1)")
fig.tight_layout()
fig.savefig(OUT_DIR / "q1_5_association.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")