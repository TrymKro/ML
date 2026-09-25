"""Ex5 Q5.3 - RBF boundary shape as gamma grows.

Uses the 2D projection from Q5.1 to fit RBF SVMs at four gamma values, and
shows how the boundary changes from smooth to wiggly as the "reach" of each
kernel distance shrinks.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import pandas as pd

from src.split import make_split
from exercise5.plot_utils import decision_surface

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X2_svm = pd.read_csv(OUT_DIR / "q5_1_train2d.csv")
_, _, _, _, _, _, _, _, y_svm, _ = make_split()

gammas = [0.001, 0.01, 0.1, 1.0]
rows = []
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, g in zip(axes, gammas):
    model = SVC(kernel="rbf", C=1.0, gamma=g)
    model.fit(X2_svm.to_numpy(), y_svm.to_numpy())
    n_sv = int(model.n_support_.sum())
    rows.append({"gamma": g, "n_support": n_sv, "sv_share": n_sv / len(X2_svm)})
    decision_surface(ax, model, X2_svm, y_svm, f"gamma = {g}  ({n_sv} SVs)", show_sv=True)

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q5_3_rbf_boundaries.csv", index=False)
print(res.round(4).to_string(index=False))

fig.suptitle("RBF decision boundary vs. gamma (training points in 2D, SV highlighted)")
fig.tight_layout()
fig.savefig(OUT_DIR / "q5_3_rbf_boundaries.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")