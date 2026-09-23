"""Ex1 Q1.8 - The leak-free pipeline.

Shows, with a concrete example, what happens if a preprocessing step is fit
on the whole dataset before the split instead of on the training fold, then
applies the actual pipeline (impute -> rare group -> one-hot, plus
winsorize + standardize for the numeric branch) to hold-out data.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_raw
from src.preprocessing import MIN_COUNT, tree_pipeline, svm_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, X_test, y_train, y_val, y_test, *_ = make_split()

# ---------------------------------------------------------------------------
# Leak demo 1: a scaler fit on everything vs. on the training fold.
# ---------------------------------------------------------------------------
full = pd.concat([X_train, X_val, X_test])

sc_full = StandardScaler().fit(full[NUMERIC_COLUMNS])
sc_train = StandardScaler().fit(X_train[NUMERIC_COLUMNS])

leak1 = pd.DataFrame(
    {
        "feature": NUMERIC_COLUMNS,
        "mean_fit_on_all": sc_full.mean_.round(2),
        "mean_fit_on_train": sc_train.mean_.round(2),
        "std_fit_on_all": sc_full.scale_.round(2),
        "std_fit_on_train": sc_train.scale_.round(2),
    }
).assign(shift=lambda d: (d["mean_fit_on_all"] - d["mean_fit_on_train"]).abs().round(2))
print("scaler fitted on all rows vs. training fold only:")
print(leak1.to_string(index=False))

# One validation row gets two different z-scores depending on where the
# scaler was fit.
row = X_val.iloc[0][NUMERIC_COLUMNS].to_numpy().reshape(1, -1)
print("\nz-score of the first validation row (all-data scaler):",
      sc_full.transform(row).round(3))
print("z-score of the same row   (train scaler):        ",
      sc_train.transform(row).round(3))

# ---------------------------------------------------------------------------
# Leak demo 2: the rare-category threshold applied to the wrong population.
# The threshold is 0.5% of the population it is fit on.
# ---------------------------------------------------------------------------
leak2 = []
for label, data, n in [
    ("train only", X_train, len(X_train)),
    ("full data", full, len(full)),
]:
    min_count = max(1, int(0.005 * n))
    kept = {}
    for col in CATEGORICAL_COLUMNS:
        counts = data[col].fillna("Unknown").value_counts()
        kept[col] = int((counts >= min_count).sum())
    leak2.append({"fit_on": label, "min_count": min_count, **kept})
leak2 = pd.DataFrame(leak2).set_index("fit_on")
print("\ncategories kept when the 0.5% threshold is fit on:")
print(leak2.to_string())

# ---------------------------------------------------------------------------
# The actual pipeline: fit on the training fold, transform the rest.
# ---------------------------------------------------------------------------
pipe = tree_pipeline()
pipe.fit(X_train, y_train)
Xt = pipe.named_steps["pre"].transform(X_test)
svm = svm_pipeline({"kernel": "linear", "C": 1.0})
svm.fit(X_train, y_train)
Xs = svm.named_steps["pre"].transform(X_test)
print(f"\ntree pipeline:  {len(NUMERIC_COLUMNS)} numeric + {Xt.shape[1] - len(NUMERIC_COLUMNS)} dummies"
      f" on test ({Xt.shape[1]} cols total)")
print(f"svm pipeline:   {Xs.shape[1]} cols on test (dummies + scaled numerics)")

leak2.to_csv(OUT_DIR / "q1_8_leak.csv")
print(f"\nsaved outputs -> {OUT_DIR}")