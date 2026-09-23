"""Ex2 Q2.5 - the class_weight='balanced' tree.

Compares the unweighted and the balanced tree on the validation set with
precision/recall/F1 for the minority (income > 50K) class.
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
from src.preprocessing import tree_pipeline
from src.split import make_split

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, _, *_ = make_split()

rows = []
for label, kwargs in [("default", {}), ("balanced", {"class_weight": "balanced"})]:
    m = tree_pipeline(**kwargs)
    m.fit(X_train, y_train)
    s = evaluate(y_val, m.predict(X_val), m.predict_proba(X_val)[:, 1])
    s = {"model": label, **s}
    rows.append(s)

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT_DIR / "q2_5_weight.csv", index=False)
print(res.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
means = res.set_index("model")[["precision", "recall", "f1"]]
means.T.plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
ax.set_ylim(0, 1)
ax.set_ylabel("score")
ax.set_xticklabels(["precision", "recall", "F1"], rotation=0)
ax.legend(title=None)
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_5_weight.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")