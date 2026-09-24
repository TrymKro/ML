"""Ex4 Q4.1 - decision tree vs SVM head to head.

Cross-validation (mean +- std) on the training data plus the validation
set for the best tree (min_samples_leaf=50, Q2.4) and the best linear SVM
(C=0.1, Q3.3). F2 is included as a second cost-sensitive metric for Q4.2,
alongside F1.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
from sklearn.metrics import fbeta_score, roc_curve
import pandas as pd

from src.metrics import cross_validate, evaluate
from src.preprocessing import svm_pipeline, tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, folds, X_svm, y_svm, svm_folds = make_split()

# Best tree from Q2.4 and best linear SVM from Q3.3 (C=0.5, selected on validation).
TREE_PARAMS = {"min_samples_leaf": 50}
SVM_PARAMS = {"kernel": "linear", "C": 0.5}

tree_cv = cross_validate(X_train, y_train, folds, lambda: tree_pipeline(**TREE_PARAMS))
svm_cv = cross_validate(X_svm, y_svm, svm_folds, lambda: svm_pipeline(SVM_PARAMS))
tree_cv.to_csv(OUT_DIR / "q4_1_tree_cv.csv", index=False)
svm_cv.to_csv(OUT_DIR / "q4_1_svm_cv.csv", index=False)

print("cross-validation (mean +- std):")
for name, cv in [("tree", tree_cv), ("svm", svm_cv)]:
    print(f"  {name:5s} acc {cv.accuracy.mean():.4f} +- {cv.accuracy.std():.4f}   "
          f"f1 {cv.f1.mean():.4f} +- {cv.f1.std():.4f}   "
          f"auc {cv.roc_auc.mean():.4f} +- {cv.roc_auc.std():.4f}")


def score_on_validation(model, fit_from, y_fit):
    model.fit(fit_from, y_fit)
    score = (model.decision_function(X_val)
             if hasattr(model.named_steps["model"], "decision_function")
             else model.predict_proba(X_val)[:, 1])
    s = evaluate(y_val, model.predict(X_val), score)
    s["f2"] = fbeta_score(y_val, model.predict(X_val), beta=2)
    return s


tree_val = pd.DataFrame([score_on_validation(tree_pipeline(**TREE_PARAMS), X_train, y_train)])
svm_val = pd.DataFrame([score_on_validation(svm_pipeline(SVM_PARAMS), X_svm, y_svm)])
res = pd.concat(
    [tree_val.assign(model="tree"), svm_val.assign(model="svm")],
    ignore_index=True,
)
res = res[["model"] + [c for c in res.columns if c != "model"]]
res.round(4).to_csv(OUT_DIR / "q4_1_validation.csv", index=False)
print("\nvalidation set:")
print(res.round(4).to_string(index=False))

# Grouped bars and overlaid ROC curves for the two models.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_cols = [c for c in ["accuracy", "balanced_acc", "f1", "f2", "roc_auc"] if c in res.columns]
res.set_index("model")[plot_cols].T.plot.bar(ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("score")

for name, model, fit_from, y_fit, auc in [
    ("tree", tree_pipeline(**TREE_PARAMS), X_train, y_train, tree_val["roc_auc"].iloc[0]),
    ("svm", svm_pipeline(SVM_PARAMS), X_svm, y_svm, svm_val["roc_auc"].iloc[0]),
]:
    model.fit(fit_from, y_fit)
    score = (model.decision_function(X_val)
             if hasattr(model.named_steps["model"], "decision_function")
             else model.predict_proba(X_val)[:, 1])
    fpr, tpr, _ = roc_curve(y_val, score)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("false positive rate")
axes[1].set_ylabel("true positive rate")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q4_1_compare.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")