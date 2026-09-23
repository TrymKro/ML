"""Ex2 Q2.2 - feature importances of the default tree.

Fits the default tree on the full training set and reports the importance
of each original feature, aggregating the one-hot dummies back to their
source column.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
import pandas as pd

from src.data import NUMERIC_COLUMNS
from src.preprocessing import CAT_ENCODE, tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, _, _, y_train, _, _, _, *_ = make_split()

model = tree_pipeline()
model.fit(X_train, y_train)
pre = model.named_steps["pre"]
ohe = pre.named_transformers_["cat"].named_steps["onehot"]
names = list(NUMERIC_COLUMNS) + list(ohe.get_feature_names_out(CAT_ENCODE))
base = [n.split("__")[-1].split("_")[0] for n in names]
imps = pd.DataFrame({"feature": base, "importance": model.named_steps["model"].feature_importances_})
imps = imps.groupby("feature")["importance"].sum().sort_values(ascending=False)

imps.round(4).to_csv(OUT_DIR / "q2_2_importance.csv")
print("feature importances (aggregated from dummies):")
print(imps.round(4).to_string())
print("\ntop 5:")
print(imps.head(5).round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
imps.sort_values().plot.barh(ax=ax, color="#4c72b0")
ax.set_xlabel("importance")
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_2_importance.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")