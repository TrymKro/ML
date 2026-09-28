"""Ex1 Q1.5 - Association between the features.

Builds a single 14x14 association matrix for the mixed-type data. Pearson |r|
for numeric-numeric pairs and Cramer's V (the measure the brief suggests) for
categorical-categorical pairs. A numeric-categorical pair is scored as the
largest Pearson |r| between the numeric and any one-hot level of the
categorical, so no statistic beyond Pearson is needed for the mixed cells.
All values lie in [0, 1] so one heatmap works for every combination.
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

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_all
from src.preprocessing import MIN_COUNT

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = load_all()

# Same cleaning as the modelling pipeline: 'Unknown' missing markers (Q1.1)
# and the rare-category grouping (Q1.3).
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].fillna("Unknown")
for col in CATEGORICAL_COLUMNS:
    counts = df[col].value_counts()
    keep = counts[counts >= MIN_COUNT].index
    df[col] = df[col].where(df[col].isin(keep), "other")

FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


# The one categorical with a natural order (Q1.4). Scored as an ordered
# variable as well as level-by-level, so the 1:1 with education-num shows up.
ORDINAL_ORDER = {
    "education": [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th",
        "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm",
        "Bachelors", "Masters", "Prof-school", "Doctorate",
    ],
}


def cramers_v(a, b):
    """Cramer's V between two categorical series."""
    tab = pd.crosstab(a, b)
    chi2, *_ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / n / min(r - 1, k - 1)))


def max_level_r(num, cat):
    """Largest Pearson |r| between a numeric series and any encoding of a categorical.

    Scored against the one-hot level dummies, and against the ordered code as
    well when the categorical has a natural order, so the 1:1 between
    education and education-num shows up as 1.00 instead of hiding in a level.
    """
    dummies = pd.get_dummies(df[cat], prefix=cat, dtype=float)
    best = max(abs(dummies[c].corr(df[num])) for c in dummies.columns)
    if cat in ORDINAL_ORDER:
        codes = df[cat].map({lvl: i for i, lvl in enumerate(ORDINAL_ORDER[cat])})
        best = max(best, abs(codes.corr(df[num])))
    return float(best)


assoc = pd.DataFrame(np.eye(len(FEATURES)), index=FEATURES, columns=FEATURES)
for i, a in enumerate(FEATURES):
    for j, b in enumerate(FEATURES):
        if i <= j:
            continue
        if a in NUMERIC_COLUMNS and b in NUMERIC_COLUMNS:
            v = abs(df[a].corr(df[b]))
        elif a in CATEGORICAL_COLUMNS and b in CATEGORICAL_COLUMNS:
            v = cramers_v(df[a], df[b])
        else:
            num, cat = (a, b) if a in NUMERIC_COLUMNS else (b, a)
            v = max_level_r(num, cat)
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