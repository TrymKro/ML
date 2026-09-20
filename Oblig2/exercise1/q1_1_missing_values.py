"""Ex1 Q1.1 - Missing values.

Prints the missing-value counts per feature, checks whether the missingness
is concentrated in subgroups, and saves a figure for the report.
"""

import sys
from pathlib import Path

# Make the code/ folder visible so `from src.data import ...` works when this
# script is run directly (python exercise1/q1_1_missing_values.py).
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import pandas as pd

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET, load_raw

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

train, test = load_raw()
print(f"train shape: {train.shape}, test shape: {test.shape}")


# --------------------------------------------------------------------------
# how many missing values does each feature have?
# --------------------------------------------------------------------------
def missing_summary(df, name):
    """Return a frame with missing counts + percentages per column."""
    counts = df.isna().sum()
    summary = pd.DataFrame(
        {
            "feature": counts.index,
            "n_missing": counts.values,
            "pct": (counts.values / len(df) * 100).round(2),
        }
    ).sort_values("n_missing", ascending=False)
    print(f"\n=== missing values in {name} ===")
    print(summary.to_string(index=False))
    return summary


train_missing = missing_summary(train, "train")
missing_summary(test, "test")

# Only workclass, occupation and native-country have missing values ('?').
assert set(train_missing[train_missing.n_missing > 0]["feature"]) == {
    "workclass",
    "occupation",
    "native-country",
}


# --------------------------------------------------------------------------
# is missingness random, or tied to specific subgroups?
# --------------------------------------------------------------------------
print("\n=== missingness pattern (train) ===")

# Do missing values overlap between features (rows with several missing)?
pattern_cols = ["workclass", "occupation", "native-country"]
miss = train[pattern_cols].isna().astype(int)
cooccur = miss.T.dot(miss)
print("\nco-occurrence of missing values (rows where both are missing):")
print(cooccur.to_string())

# The obvious pairing to check: does a missing occupation go together
# with a missing workclass?
ct = pd.crosstab(
    train["occupation"].isna(),
    train["workclass"].isna(),
).rename(index={False: "occupation present", True: "occupation missing"},
         columns={False: "workclass present", True: "workclass missing"})
print("\ncrosstab: occupation x workclass missingness")
print(ct.to_string())

# MCAR sanity check: compare the numeric features of rows with a missing
# occupation vs. rows without. Strong differences suggest the data is NOT
# missing completely at random (MCAR) but more likely missing at random.
odd_rows = train["occupation"].isna()
print("\nmedian numeric features: missing-occupation rows vs. complete rows")
compared = pd.DataFrame(
    {
        "missing_occupation": train.loc[odd_rows, NUMERIC_COLUMNS].median(),
        "complete": train.loc[~odd_rows, NUMERIC_COLUMNS].median(),
    }
).round(1)
print(compared.to_string())

# Is the missing share different between low and high income? If so, the
# missingness correlates with the target, which matters downstream.
print("\nmissing workclass/occupation by income class:")
for col in ["workclass", "occupation"]:
    tab = pd.crosstab(train[TARGET], train[col].isna(), normalize="index")
    print(f"- {col}\n  {tab.round(4).to_string()}")

# Save the numbers for the report and make the two figures.
train_missing.to_csv(OUT_DIR / "q1_1_missing_counts.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Bar chart of the missing counts.
axes[0].barh(train_missing["feature"], train_missing["n_missing"], color="#4c72b0")
axes[0].set_title("Missing values per feature (train)")
axes[0].set_xlabel("n rows missing")

# Visualised co-occurrence matrix.
im = axes[1].imshow(cooccur.values, cmap="Blues")
axes[1].set_xticks(range(len(pattern_cols)), pattern_cols, rotation=45)
axes[1].set_yticks(range(len(pattern_cols)), pattern_cols)
axes[1].set_title("Co-occurrence of missing values")
for i in range(len(cooccur)):
    for j in range(len(cooccur)):
        axes[1].text(j, i, int(cooccur.values[i, j]), ha="center", va="center")
fig.colorbar(im, ax=axes[1], shrink=0.8)

fig.tight_layout()
fig.savefig(OUT_DIR / "q1_1_missing_values.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")