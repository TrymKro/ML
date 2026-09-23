"""Ex1 Q1.6 - Association of each feature with the target.

Numeric features are tested against the binary target with an ANOVA F-test
(equivalent to point-biserial correlation in rank), categorical features with
the chi-square test. Because every p-value is essentially zero, the features
are ranked by a comparable effect size instead: |point-biserial r| for the
numerics and Cramer's V (here sqrt(chi2/n), since the target has two levels)
for the categoricals, both on a 0-1 scale.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from sklearn.feature_selection import f_classif

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.preprocessing import MIN_COUNT
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, _, _, y_train, _, _, _, _, _, _ = make_split()

# Same cleaning as the modelling pipeline: 'Unknown' missing markers (Q1.1)
# and the rare-category grouping (Q1.3).
for col in ["workclass", "occupation", "native-country"]:
    X_train[col] = X_train[col].fillna("Unknown")
for col in CATEGORICAL_COLUMNS:
    counts = X_train[col].value_counts()
    keep = counts[counts >= MIN_COUNT].index
    X_train[col] = X_train[col].where(X_train[col].isin(keep), "other")

rows = []
for col in NUMERIC_COLUMNS:
    f_stat, p_val = f_classif(X_train[[col]].to_numpy(), y_train)
    r, _ = pearsonr(X_train[col], y_train)
    rows.append(
        {
            "feature": col,
            "type": "numeric",
            "statistic": round(float(f_stat[0]), 3),
            "p_value": float(p_val[0]),
            "effect": abs(r),
        }
    )
for col in CATEGORICAL_COLUMNS:
    tab = pd.crosstab(X_train[col], y_train)
    chi2, p_val, *_ = chi2_contingency(tab)
    effect = float(np.sqrt(chi2 / tab.to_numpy().sum()))
    rows.append(
        {
            "feature": col,
            "type": "categorical",
            "statistic": round(float(chi2), 1),
            "p_value": float(p_val),
            "effect": effect,
        }
    )

res = pd.DataFrame(rows)
res["log10p"] = -np.log10(res["p_value"].clip(lower=1e-300))
res = res.sort_values(["effect", "log10p"], ascending=False).reset_index(drop=True)
res.to_csv(OUT_DIR / "q1_6_association_tests.csv", index=False)

res["effect"] = res["effect"].round(3)
res["log10p"] = res["log10p"].round(1)
print(res.to_string(index=False))
print("\ntop 8 features by association strength:")
print(res.head(8).to_string(index=False))