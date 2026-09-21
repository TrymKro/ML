"""Ex1 Q1.3 - Cardinality of the categorical features.

Counts how many categories each categorical feature has on the training
set, then folds the rare ones into an 'other' bucket by frequency. Missing
values are filled with 'Unknown' first, as decided in Q1.1.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd

from src.data import CATEGORICAL_COLUMNS, load_raw

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train, test = load_raw()

# Q1.1 decision: '?' becomes an explicit 'Unknown' category, so the
# missing-value signal survives into the encoding step.
for col in ["workclass", "occupation", "native-country"]:
    train[col] = train[col].fillna("Unknown")


def group_rare(series, min_count):
    """Replace categories seen in fewer than min_count rows with 'other'."""
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), "other")


# How many categories survive at a few thresholds? The threshold is picked
# on the training rows only, so it can be applied to test later without
# leaking anything.
thresholds = [0.001, 0.002, 0.003, 0.005, 0.01]  # 0.1% ... 1% of the training rows
sens = pd.DataFrame(
    {
        f"{t * 100:.1f}%": {
            col: len(group_rare(train[col], max(1, int(t * len(train)))).unique())
            for col in CATEGORICAL_COLUMNS
        }
        for t in thresholds
    }
)
sens.index.name = "feature"
print("categories kept per threshold:")
print(sens.to_string())

thr = 0.005  # chosen: keep categories with at least 0.5% support in train
min_count = max(1, int(thr * len(train)))
print(f"\nchosen threshold: >= {min_count} rows ({thr * 100:.1f}% of train)")

rows = []
for col in CATEGORICAL_COLUMNS:
    before = train[col].nunique()
    grouped = group_rare(train[col], min_count)
    n_other = (grouped == "other").sum()
    rows.append(
        {
            "feature": col,
            "n_before": before,
            "n_after": grouped.nunique(),
            "rows_in_other": n_other,
            "pct_in_other": round(n_other / len(train) * 100, 2),
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "q1_3_cardinality.csv", index=False)
print("\nsummary (0.5% threshold):")
print(summary.to_string(index=False))

# Detailed counts for the high-cardinality examples after grouping.
for col in ["native-country", "occupation"]:
    grouped = group_rare(train[col], min_count)
    print(f"\n{col}: {train[col].nunique()} categories -> {grouped.nunique()} after grouping")
    print(grouped.value_counts().sort_index().to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col in zip(axes, ["native-country", "occupation"]):
    grouped = group_rare(train[col], min_count)
    counts = grouped.value_counts().sort_values()
    other = counts.pop("other") if "other" in counts.index else None
    counts.plot.barh(ax=ax, color="#4c72b0")
    if other is not None:
        ax.barh(len(counts), other, color="#c44e52", label=f"other ({int(other)} rows)")
        ax.legend()
    ax.set_title(col)

fig.tight_layout()
fig.savefig(OUT_DIR / "q1_3_cardinality.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")