"""Ex5 Q5.4 - decision tree boundary as max depth grows.

Using the same 2D projection, four trees of increasing depth show the
transition from coarse axis-aligned splits to a fine-grained, overfitted
tessellation.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from src.metrics import evaluate
from src.split import make_split
from exercise5.plot_utils import decision_surface

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X2_svm = pd.read_csv(OUT_DIR / "q5_1_train2d.csv")
X2_val = pd.read_csv(OUT_DIR / "q5_1_val2d.csv")
_, _, _, _, y_val, _, _, _, y_svm, _ = make_split()

depths = [2, 6, 10, 15]
rows = []
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, d in zip(axes, depths):
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X2_svm.to_numpy(), y_svm.to_numpy())
    s = evaluate(y_val.to_numpy(), model.predict(X2_val.to_numpy()),
                 model.predict_proba(X2_val.to_numpy())[:, 1])
    rows.append({"max_depth": d, **{k: s[k] for k in ["accuracy", "f1", "roc_auc"]}})
    decision_surface(ax, model, X2_val, y_val, f"depth = {d}")

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q5_4_tree_boundaries.csv", index=False)
print(res.round(4).to_string(index=False))

fig.suptitle("Decision tree boundary vs. max depth (validation points overlaid)")
fig.tight_layout()
fig.savefig(OUT_DIR / "q5_4_tree_boundaries.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")