"""Ex4 Q4.4 - final held-out test set evaluation.

Fits the two models we deploy (the hand-tuned tree from Q2.4 and the
grid-searched RBF SVM from Q4.3) and scores them on the test set, which has
been unused for model selection. The grid's own tree pick is scored as well,
purely as a check on the Q4.3 decision, and is not used to choose anything.
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

# Deployed settings: the tree is the Q2.4 winner (the Q4.3 grid found nothing
# better, see the report), the SVM is the Q4.3 grid winner.
TREE_PARAMS = {"min_samples_leaf": 50}
TREE_PARAMS_GRID = {"max_depth": 12, "min_samples_leaf": 25}
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": 0.1}

X_train, _, X_test, y_train, _, y_test, _, X_svm, y_svm, _ = make_split()

rows = []
for name, model, fit_from, y_fit in [
    ("tree", tree_pipeline(**TREE_PARAMS), X_train, y_train),
    ("tree (Q4.3 grid pick)", tree_pipeline(**TREE_PARAMS_GRID), X_train, y_train),
    ("svm", svm_pipeline(SVM_PARAMS), X_svm, y_svm),
]:
    model.fit(fit_from, y_fit)
    score = (model.decision_function(X_test)
             if hasattr(model.named_steps["model"], "decision_function")
             else model.predict_proba(X_test)[:, 1])
    s = evaluate(y_test, model.predict(X_test), score)
    s["train_acc"] = model.score(fit_from, y_fit)
    rows.append({"model": name, **s})

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q4_4_test.csv", index=False)
print(res.round(4).to_string(index=False))

# Ranks and agreement with the validation ranking.
rank_val = ["tree", "svm"]
print("\nF1 order on test set :", res.sort_values("f1", ascending=False)["model"].tolist())
print("F1 order on validation:", rank_val)