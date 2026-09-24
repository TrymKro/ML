"""Ex4 Q4.4 - final held-out test set evaluation.

Fits the grid-searched best decision tree (full training set) and the
grid-searched best SVM (training sub-sample), then scores both on the test
set that has been unused for model selection.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.metrics import evaluate
from src.preprocessing import svm_pipeline, tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Best settings from Q4.3.
TREE_PARAMS = {"max_depth": 12, "min_samples_leaf": 25}
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": 0.1}

X_train, _, X_test, y_train, _, y_test, _, X_svm, y_svm, _ = make_split()

rows = []
for name, model, fit_from, y_fit in [
    ("tree", tree_pipeline(**TREE_PARAMS), X_train, y_train),
    ("svm", svm_pipeline(SVM_PARAMS), X_svm, y_svm),
]:
    model.fit(fit_from, y_fit)
    score = (model.decision_function(X_test)
             if hasattr(model.named_steps["model"], "decision_function")
             else model.predict_proba(X_test)[:, 1])
    s = evaluate(y_test, model.predict(X_test), score)
    rows.append({"model": name, **s})

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q4_4_test.csv", index=False)
print(res.round(4).to_string(index=False))

# Ranks and agreement with the validation ranking.
rank_val = ["tree", "svm"]
print("\nF1 order on test set :", res.sort_values("f1", ascending=False)["model"].tolist())
print("F1 order on validation:", rank_val)