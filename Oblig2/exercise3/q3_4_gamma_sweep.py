"""Ex3 Q3.4 - gamma sweep for the RBF kernel.

Keeps the RBF kernel at the default cost C=1 and sweeps gamma from 0.0001
to 10, reporting 5-fold CV and validation scores.
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

g_vals = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
rows = []
roc_curves = {}
for g in g_vals:
    kwargs = {"kernel": "rbf", "C": 1.0, "gamma": g}
    cv = cross_validate(X_svm, y_svm, svm_folds, lambda k=kwargs: svm_pipeline(k))
    model = svm_pipeline(kwargs)
    model.fit(X_svm, y_svm)
    score = model.decision_function(X_val)
    n_support = int(model.named_steps["model"].n_support_.sum())
    val = evaluate(y_val, model.predict(X_val), score)
    rows.append(
        {
            "gamma": g,
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
    roc_curves[g] = roc_curve(y_val, score)

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q3_4_gamma_sweep.csv", index=False)
print(res.round(4).to_string(index=False))
print("\nbest gamma by validation AUC:", res.sort_values("val_auc", ascending=False).iloc[0]["gamma"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].errorbar(res["gamma"], res["cv_acc"], yerr=res["cv_acc_std"], fmt="o-")
axes[0].set_xscale("log")
axes[0].set_xlabel("gamma")
axes[0].set_ylabel("CV accuracy")
axes[1].errorbar(res["gamma"], res["cv_f1"], yerr=res["cv_f1_std"], fmt="o-", label="CV F1")
axes[1].plot(res["gamma"], res["val_auc"], "s--", color="#dd8452", label="val AUC")
axes[1].set_xscale("log")
axes[1].set_xlabel("gamma")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_4_gamma_sweep.png", dpi=150)

# ROC curves of the four middle gammas, too many lines otherwise.
fig, ax = plt.subplots(figsize=(6, 5))
for g in [0.001, 0.01, 0.1, 0.5]:
    fpr, tpr, _ = roc_curves[g]
    auc = res.loc[res["gamma"] == g, "val_auc"].iloc[0]
    ax.plot(fpr, tpr, label=f"gamma={g} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_4_roc.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")