"""Ex4 Q4.3 - GridSearchCV on the training set.

Jointly tunes the decision tree (max_depth, min_samples_leaf) and, on the
SVM sub-sample, the SVM (kernel, C and gamma). F1 is used as the scoring
metric because the classes are imbalanced. The grid-search picks are then
confirmed on the untouched validation set against the earlier hand-tuned
bests.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
import pandas as pd

from src.metrics import evaluate
from src.preprocessing import svm_pipeline, tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, folds, X_svm, y_svm, svm_folds = make_split()


def grid_fit(pipe, grid, fit_from, y_fit, folds_for_cv):
    from sklearn.model_selection import GridSearchCV

    gs = GridSearchCV(pipe, grid, scoring="f1", cv=folds_for_cv, n_jobs=1)
    gs.fit(fit_from, y_fit)
    return gs


# Decision tree search on the full training set.
tree_grid = [
    {"model__max_depth": [6, 9, 12], "model__min_samples_leaf": [10, 25, 50]}
]
tree_gs = grid_fit(tree_pipeline(), tree_grid, X_train, y_train, folds)
print("grid-search decision tree:")
print("  best params:", tree_gs.best_params_)
print(f"  best CV score (f1): {tree_gs.best_score_:.4f} +- {tree_gs.cv_results_['std_test_score'][tree_gs.best_index_]:.4f}")
tree_best = tree_gs.best_estimator_

# SVM search on the 5,000-row sub-sample.
svm_grid = [
    {"model__kernel": ["linear"], "model__C": [0.1, 0.5, 1.0]},
    {"model__kernel": ["rbf"], "model__C": [0.5, 1.0], "model__gamma": [0.05, 0.1, 0.2]},
]
svm_gs = grid_fit(svm_pipeline({}), svm_grid, X_svm, y_svm, svm_folds)
print("\ngrid-search SVM:")
print("  best params:", svm_gs.best_params_)
print(f"  best CV score (f1): {svm_gs.best_score_:.4f} +- {svm_gs.cv_results_['std_test_score'][svm_gs.best_index_]:.4f}")
svm_best = svm_gs.best_estimator_

# Validation confirmation, against the hand-tuned baselines.
def eval_on_val(model):
    score = (model.decision_function(X_val)
             if hasattr(model.named_steps["model"], "decision_function")
             else model.predict_proba(X_val)[:, 1])
    return pd.Series(evaluate(y_val, model.predict(X_val), score))


rows = [
    {"config": "grid tree", **eval_on_val(tree_best).to_dict()},
    {"config": "q2.4 tree (msl=50)", **eval_on_val(tree_pipeline(min_samples_leaf=50).fit(X_train, y_train)).to_dict()},
    {"config": "grid svm", **eval_on_val(svm_best).to_dict()},
]
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q4_3_validation.csv", index=False)
print("\nvalidation confirmation:")
print(res.round(4).to_string(index=False))

# CV summary of the four candidates.
summary = pd.DataFrame(
    {
        "model": ["tree", "svm"],
        "grid": [tree_gs.best_score_, svm_gs.best_score_],
        "grid_std": [tree_gs.cv_results_["std_test_score"][tree_gs.best_index_],
                     svm_gs.cv_results_["std_test_score"][svm_gs.best_index_]],
        "params": [str(tree_gs.best_params_), str(svm_gs.best_params_)],
    }
)
summary.round(4).to_csv(OUT_DIR / "q4_3_grid_cv.csv", index=False)
print("\ngrid CV summary:")
print(summary.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(summary["model"], summary["grid"], yerr=summary["grid_std"],
       color=["#4c72b0", "#dd8452"], capsize=5)
ax.set_ylabel("best CV F1")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT_DIR / "q4_3_grid_cv.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")