"""Ex2 Q2.3 - tree depth sweep.

Varies max_depth from 2 to 15, reporting accuracy and F1 (mean +- std from
the 5-fold CV) plus the training accuracy so the train/validation gap can be
read off, and scores the deepening trees on the validation set.
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

depths = range(2, 16)
rows = []
for d in depths:
    cv = cross_validate(X_train, y_train, folds, lambda d=d: tree_pipeline(max_depth=d), want_train=True)
    rows.append(
        {
            "max_depth": d,
            "cv_acc_mean": cv.accuracy.mean(),
            "cv_acc_std": cv.accuracy.std(),
            "cv_f1_mean": cv.f1.mean(),
            "cv_f1_std": cv.f1.std(),
            "cv_train_acc_mean": cv.train_acc.mean(),
        }
    )
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q2_3_depth.csv", index=False)
print(res.round(4).to_string(index=False))

# Validation scores of each depth, to confirm the CV picture.
best_d = res.sort_values("cv_f1_mean", ascending=False).iloc[0]["max_depth"]
print(f"\nbest depth by CV F1: {int(best_d)}")
for d in depths:
    m = tree_pipeline(max_depth=d)
    m.fit(X_train, y_train)
    s = evaluate(y_val, m.predict(X_val), m.predict_proba(X_val)[:, 1])
    print(f"depth {d:2d}  val acc {s['accuracy']:.4f}  val f1 {s['f1']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = res["max_depth"]
axes[0].plot(x, res["cv_acc_mean"], "o-", label="CV accuracy")
axes[0].plot(x, res["cv_train_acc_mean"], "s--", label="train accuracy")
axes[0].set_xlabel("max depth")
axes[0].set_ylabel("accuracy")
axes[0].legend()
axes[1].plot(x, res["cv_f1_mean"], "o-")
axes[1].set_xlabel("max depth")
axes[1].set_ylabel("CV F1")
axes[1].fill_between(x, res["cv_f1_mean"] - res["cv_f1_std"],
                     res["cv_f1_mean"] + res["cv_f1_std"], alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_3_depth.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")