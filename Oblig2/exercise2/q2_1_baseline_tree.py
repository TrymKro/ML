"""Ex2 Q2.1 - baseline decision tree, default parameters.

5-fold cross-validation on the training set (accuracy and balanced
accuracy, mean +- std across the folds), then a model fitted on the full
training set and scored on the untouched validation set.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics import cross_validate, evaluate
from src.preprocessing import tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, folds, *_ = make_split()

cv = cross_validate(X_train, y_train, folds, lambda: tree_pipeline())
cv.to_csv(OUT_DIR / "q2_1_cv.csv", index=False)

print("default decision tree, 5-fold CV on the training set:")
print(cv.round(4).to_string(index=False))
print("\nmean +- std:")
print(f"  accuracy:       {cv.accuracy.mean():.4f} +- {cv.accuracy.std():.4f}")
print(f"  balanced acc:   {cv.balanced_acc.mean():.4f} +- {cv.balanced_acc.std():.4f}")

# Final model on all of the training data, scored on the validation set.
model = tree_pipeline()
model.fit(X_train, y_train)
pred = model.predict(X_val)
score = model.predict_proba(X_val)[:, 1]
val = pd.DataFrame([evaluate(y_val, pred, score)])
val.to_csv(OUT_DIR / "q2_1_validation.csv", index=False)
print("\nvalidation set:")
print(val.round(4).to_string(index=False))