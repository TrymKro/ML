"""Ex3 Q3.2 - linear vs RBF kernel.

Both kernels with the default cost C=1 and (for RBF) the default gamma,
compared by 5-fold CV on the sub-sample and by their ROC curves on the
validation set.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

from src.metrics import cross_validate, evaluate
from src.preprocessing import svm_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

_, _, X_val, _, _, y_val, _, X_svm, y_svm, svm_folds = make_split()

rows = []
fig, ax = plt.subplots(figsize=(6, 5))
for kernel in ["linear", "rbf"]:
    kwargs = {"kernel": kernel, "C": 1.0}
    cv = cross_validate(X_svm, y_svm, svm_folds, lambda k=kwargs: svm_pipeline(k))
    model = svm_pipeline(kwargs)
    model.fit(X_svm, y_svm)
    score = model.decision_function(X_val)
    pred = model.predict(X_val)
    val = evaluate(y_val, pred, score)
    rows.append(
        {
            "kernel": kernel,
            "cv_acc": cv.accuracy.mean(),
            "cv_acc_std": cv.accuracy.std(),
            "cv_bal_acc": cv.balanced_acc.mean(),
            "cv_f1": cv.f1.mean(),
            "val_acc": val["accuracy"],
            "val_f1": val["f1"],
            "val_auc": val["roc_auc"],
        }
    )
    fpr, tpr, _ = roc_curve(y_val, score)
    ax.plot(fpr, tpr, label=f"{kernel} (AUC = {val['roc_auc']:.3f})")

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q3_2_kernels.csv", index=False)
print(res.round(4).to_string(index=False))

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_2_kernels.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")