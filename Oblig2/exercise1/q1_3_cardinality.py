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

from src.data import CATEGORICAL_COLUMNS, load_all
from src.preprocessing import MIN_COUNT

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = load_all()

# Q1.1 decision: '?' becomes an explicit 'Unknown' category, so the
# missing-value signal survives into the encoding step.
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].fillna("Unknown")


def group_rare(series, min_count):
    """Replace categories seen in fewer than min_count rows with 'other'."""
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), "other")


# How many categories survive at a few thresholds? The threshold is picked
# on the training rows only, so it can be applied to test later without
# leaking anything.
# Absolute row counts, so the grid brackets the chosen constant MIN_COUNT
# directly: 0.1% / 0.2% / 0.3% of the merged file, the chosen 162 rows,
# then 0.5% and 1%.
thresholds = [49, 98, 147, MIN_COUNT, 244, 488]
sens = pd.DataFrame(
    {
        f"{t}": {
            col: len(group_rare(df[col], t).unique())
            for col in CATEGORICAL_COLUMNS
        }
        for t in thresholds
    }
)
sens.index.name = "feature"
print("categories kept per threshold:")
print(sens.to_string())

# The chosen operating point is the pipeline's own constant (Q1.8), not a
# percentage of this file: the model is fitted on the 29304-row training
# split, and a percentage would silently change meaning on every fold.
min_count = MIN_COUNT
print(f"\nchosen threshold: >= {min_count} rows "
      f"({min_count / len(df) * 100:.2f}% of the merged file, "
      f"{min_count / 29304 * 100:.2f}% of the training split)")

rows = []
for col in CATEGORICAL_COLUMNS:
    before = df[col].nunique()
    grouped = group_rare(df[col], min_count)
    n_other = (grouped == "other").sum()
    rows.append(
        {
            "feature": col,
            "n_before": before,
            "n_after": grouped.nunique(),
            "rows_in_other": n_other,
            "pct_in_other": round(n_other / len(df) * 100, 2),
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "q1_3_cardinality.csv", index=False)
print(f"\nsummary (MIN_COUNT={MIN_COUNT} threshold):")
print(summary.to_string(index=False))

# Detailed counts for the high-cardinality examples after grouping.
for col in ["native-country", "occupation"]:
    grouped = group_rare(df[col], min_count)
    print(f"\n{col}: {df[col].nunique()} categories -> {grouped.nunique()} after grouping")
    print(grouped.value_counts().sort_index().to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col in zip(axes, ["native-country", "occupation"]):
    grouped = group_rare(df[col], min_count)
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