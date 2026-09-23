"""Ex2 Q2.4 - min_samples_leaf and ccp_alpha sweeps.

Sweeps both regularisation knobs of the tree, reports the CV accuracy/F1 for
each value, then confirms the best settings on the validation set.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
import pandas as pd

from src.metrics import cross_validate, evaluate
from src.preprocessing import tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, folds, *_ = make_split()


def sweep(param, values):
    rows = []
    for v in values:
        cv = cross_validate(
            X_train, y_train, folds,
            lambda v=v: tree_pipeline(**{param: v}),
        )
        rows.append(
            {
                param: v,
                "cv_acc": cv.accuracy.mean(),
                "cv_acc_std": cv.accuracy.std(),
                "cv_f1": cv.f1.mean(),
                "cv_f1_std": cv.f1.std(),
            }
        )
    return pd.DataFrame(rows)


leaf_vals = [1, 3, 5, 10, 25, 50]
leaf = sweep("min_samples_leaf", leaf_vals)
print("min_samples_leaf sweep (default depth):")
print(leaf.round(4).to_string(index=False))

alpha_vals = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02]
alpha = sweep("ccp_alpha", alpha_vals)
print("\nccp_alpha sweep (default depth):")
print(alpha.round(4).to_string(index=False))

# Combine the two best readings and pick on the validation set.
best_leaf = leaf.sort_values("cv_f1", ascending=False).iloc[0]["min_samples_leaf"]
best_alpha = alpha.sort_values("cv_f1", ascending=False).iloc[0]["ccp_alpha"]
print(f"\nbest leaf {best_leaf}, best alpha {best_alpha}")

combo = []
for leaf_v in [1, int(best_leaf), 10, 25]:
    for alpha_v in [0.0, best_alpha, 0.01]:
        m = tree_pipeline(min_samples_leaf=leaf_v, ccp_alpha=alpha_v)
        m.fit(X_train, y_train)
        s = evaluate(y_val, m.predict(X_val), m.predict_proba(X_val)[:, 1])
        combo.append({"min_samples_leaf": leaf_v, "ccp_alpha": alpha_v, **s})
combo = pd.DataFrame(combo)
combo.round(4).to_csv(OUT_DIR / "q2_4_combos.csv", index=False)
print("\nvalidation scores of selected combos:")
print(combo.round(4).to_string(index=False))

leaf.round(4).to_csv(OUT_DIR / "q2_4_leaf.csv", index=False)
alpha.round(4).to_csv(OUT_DIR / "q2_4_alpha.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(leaf["min_samples_leaf"], leaf["cv_acc"], "o-", label="CV acc")
axes[0].plot(leaf["min_samples_leaf"], leaf["cv_f1"], "s--", label="CV F1")
axes[0].set_xscale("log")
axes[0].set_xlabel("min samples leaf")
axes[0].legend()
axes[1].plot(alpha["ccp_alpha"], alpha["cv_acc"], "o-", label="CV acc")
axes[1].plot(alpha["ccp_alpha"], alpha["cv_f1"], "s--", label="CV F1")
axes[1].set_xlabel("ccp alpha")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_4_pruning.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")