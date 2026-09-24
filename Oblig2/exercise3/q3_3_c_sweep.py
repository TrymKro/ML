"""Ex3 Q3.3 - cost parameter (C) sweep.

Sweeps C over six orders of magnitude for the linear kernel, reporting the
5-fold CV accuracy/F1 and the validation AUC of each model.
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
from src.preprocessing import svm_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

_, _, X_val, _, _, y_val, _, X_svm, y_svm, svm_folds = make_split()

c_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
rows = []
for c in c_vals:
    kwargs = {"kernel": "linear", "C": c}
    cv = cross_validate(X_svm, y_svm, svm_folds, lambda k=kwargs: svm_pipeline(k))
    model = svm_pipeline(kwargs)
    model.fit(X_svm, y_svm)
    n_support = int(model.named_steps["model"].n_support_.sum())
    val = evaluate(y_val, model.predict(X_val), model.decision_function(X_val))
    rows.append(
        {
            "C": c,
            "cv_acc": cv.accuracy.mean(),
            "cv_acc_std": cv.accuracy.std(),
            "cv_f1": cv.f1.mean(),
            "cv_f1_std": cv.f1.std(),
            "n_support": n_support,
            "val_acc": val["accuracy"],
            "val_f1": val["f1"],
            "val_auc": val["roc_auc"],
        }
    )

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q3_3_c_sweep.csv", index=False)
print(res.round(4).to_string(index=False))
print("\nbest C by validation AUC:", res.sort_values("val_auc", ascending=False).iloc[0]["C"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].errorbar(res["C"], res["cv_acc"], yerr=res["cv_acc_std"], fmt="o-")
axes[0].set_xscale("log")
axes[0].set_xlabel("C")
axes[0].set_ylabel("CV accuracy")
axes[1].errorbar(res["C"], res["cv_f1"], yerr=res["cv_f1_std"], fmt="o-", label="CV F1")
axes[1].plot(res["C"], res["val_auc"], "s--", color="#dd8452", label="val AUC")
axes[1].set_xscale("log")
axes[1].set_xlabel("C")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_3_c_sweep.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")