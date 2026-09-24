"""Ex3 Q3.1 - linear SVM baseline.

SVMs are quadratic-cost learners, so fitting them on all 29k training rows
is slow. Like any standard SVD/SVM exercise, we fit on a stratified
sub-sample of 5,000 training rows (kept aside in the split exactly for this)
and report the 5-fold CV on those 5,000 rows, then score the model on the
full untouched validation set.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

from src.metrics import cross_validate, evaluate
from src.preprocessing import svm_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

_, _, X_val, _, _, y_val, _, X_svm, y_svm, svm_folds = make_split()

svm_kwargs = {"kernel": "linear", "C": 1.0}

cv = cross_validate(X_svm, y_svm, svm_folds, lambda: svm_pipeline(svm_kwargs))
cv.to_csv(OUT_DIR / "q3_1_cv.csv", index=False)
print("linear SVM, 5-fold CV on the 5,000-row sub-sample:")
print(cv.round(4).to_string(index=False))
print("\nmean +- std:")
print(f"  accuracy:       {cv.accuracy.mean():.4f} +- {cv.accuracy.std():.4f}")
print(f"  balanced acc:   {cv.balanced_acc.mean():.4f} +- {cv.balanced_acc.std():.4f}")

# Per-split ROC curves, as required (the original 5 fold splits, restricted
# to the sub-sample).
fold_rocs = []
for tr, va in svm_folds:
    m = svm_pipeline(svm_kwargs)
    m.fit(X_svm.iloc[tr], y_svm.iloc[tr])
    fpr, tpr, _ = roc_curve(y_svm.iloc[va], m.decision_function(X_svm.iloc[va]))
    fold_rocs.append((fpr, tpr))

model = svm_pipeline(svm_kwargs)
model.fit(X_svm, y_svm)
score = model.decision_function(X_val)
pred = model.predict(X_val)
val = pd.DataFrame([evaluate(y_val, pred, score)])
val.to_csv(OUT_DIR / "q3_1_validation.csv", index=False)
print("\nvalidation set (full 9,769 rows):")
print(val.round(4).to_string(index=False))

fpr, tpr, _ = roc_curve(y_val, score)
fig, ax = plt.subplots(figsize=(6, 5))
for i, (f, t) in enumerate(fold_rocs):
    ax.plot(f, t, color="#c8d7ee", lw=1, label="fold ROC" if i == 0 else None)
mean_fpr = np.linspace(0, 1, 200)
mean_tpr = np.mean(
    [np.interp(mean_fpr, f, t) for f, t in fold_rocs], axis=0
)
ax.plot(mean_fpr, mean_tpr, "--", color="#4c72b0", label=f"mean fold ROC (AUC={val.roc_auc[0]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_1_roc.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")