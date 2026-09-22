"""Ex1 Q1.4 - Encoding the categorical features.

One-hot-encodes the categorical columns after the Q1.1 imputation and the
Q1.3 rare-grouping, and reports how the feature-space dimensionality
changes. education is dropped in favour of the equivalent numeric
education-num that already ships with the data.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_raw

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train, test = load_raw()


def group_rare(series, min_count):
    """Replace categories seen in fewer than min_count rows with 'other'."""
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), "other")


# Q1.1: '?' becomes an explicit 'Unknown' category.
for col in ["workclass", "occupation", "native-country"]:
    train[col] = train[col].fillna("Unknown")

# Record the raw category counts first, so we can show what the Q1.3
# grouping saves.
pre_count = {col: train[col].nunique() for col in CATEGORICAL_COLUMNS}

# Q1.3: fold rare categories into 'other' (0.5% of the training rows).
min_count = max(1, int(0.005 * len(train)))
train[CATEGORICAL_COLUMNS] = train[CATEGORICAL_COLUMNS].apply(
    lambda s: group_rare(s, min_count)
)

# education vs education-num: the text column looks like a redundant copy
# of the numeric one. Check that the mapping is deterministic (one
# education-num per education label).
check = train.groupby("education")["education-num"].nunique()
print("education labels mapped to more than one education-num:",
      int((check > 1).sum()))

# Dummy columns per feature with and without the Q1.3 grouping, both with
# 1-in-K encoding (a column must be dropped for linear models) and full.
cols = [c for c in CATEGORICAL_COLUMNS if c != "education"]
rows = []
for col in cols:
    d = len(train[col].value_counts())
    rows.append(
        {
            "feature": col,
            "dummies_after_group": d,
            "dummies_drop_first": d - 1,
        }
    )

dim_before = sum(pre_count[c] for c in cols) + pre_count["education"]
dim_after_group = sum(r["dummies_after_group"] for r in rows) + train["education"].nunique()
dim_final = sum(r["dummies_after_group"] for r in rows)  # education->education-num

res = pd.DataFrame(rows)
print("\none-hot dummies per feature (grouped at 0.5%, education excluded):")
print(res.to_string(index=False))
print(f"\ntotal one-hot dummies before grouping : {dim_before}")
print(f"total one-hot dummies after grouping  : {dim_after_group} (incl. education)")
print(f"final (education dropped, education-num kept): {dim_final}")
print(f"final feature space: {len(NUMERIC_COLUMNS)} numeric + {dim_final} dummies"
      f" = {len(NUMERIC_COLUMNS) + dim_final} columns")
print(f"with drop-first (for linear models): {len(NUMERIC_COLUMNS) + dim_final - len(res)} columns")

res.to_csv(OUT_DIR / "q1_4_encoding.csv", index=False)

# Figure: dummy columns before vs after the Q1.3 grouping, per feature.
fig, ax = plt.subplots(figsize=(9, 5))
order = res.sort_values("dummies_after_group")["feature"]
before = [pre_count[col] for col in order]
after = [train[col].nunique() for col in order]

y = range(len(order))
ax.barh([i + 0.2 for i in y], before, height=0.4, color="#4c72b0", label="before grouping")
ax.barh([i - 0.2 for i in y], after, height=0.4, color="#dd8452", label="after grouping")
ax.set_yticks(list(y), order)
ax.set_xlabel("categories (one-hot columns)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q1_4_encoding.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")