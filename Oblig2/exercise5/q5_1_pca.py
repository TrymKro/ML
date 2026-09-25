"""Ex5 Q5.1 - PCA to two dimensions and the three decision boundaries.

Fits a 2-component PCA on the preprocessed training set (standardized
numerics + one-hot categoricals), reports the variance explained by PC1 and
PC2, retrains the best decision tree and both SVMs on the sub-sample
projected to two dimensions, and plots their decision boundaries with the
validation points overlaid.
"""

import sys
from pathlib import Path

# make src/ importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # works without a display
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from src.data import NUMERIC_COLUMNS
from src.preprocessing import CAT_ENCODE, categorical_step, Winsorizer
from src.split import make_split
from exercise5.plot_utils import decision_surface

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

X_train, X_val, _, y_train, y_val, _, _, X_svm, y_svm, _ = make_split()

# Same preprocessing as the models, but with standardized numerics so that
# PCA is not dominated by the scale of one column.
pre = ColumnTransformer(
    [
        ("num", Pipeline([("winsorize", Winsorizer()), ("scale", StandardScaler())]), NUMERIC_COLUMNS),
        ("cat", categorical_step(), CAT_ENCODE),
    ]
)
pre.fit(X_train)

pca = PCA(n_components=2, random_state=42)
pca.fit(pre.transform(X_train))
var = pd.DataFrame(
    {
        "component": [1, 2],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": pca.explained_variance_ratio_.cumsum(),
    }
)
var.round(4).to_csv(OUT_DIR / "q5_1_pca_variance.csv", index=False)
print("variance explained:")
print(var.round(4).to_string(index=False))

# Project the modelling sub-sample (train) and the validation set to 2D.
def project(df):
    return pd.DataFrame(pca.transform(pre.transform(df)), columns=["PC1", "PC2"])


X2_svm, X2_val = project(X_svm), project(X_val)
X2_svm.to_csv(OUT_DIR / "q5_1_train2d.csv", index=False)
X2_val.to_csv(OUT_DIR / "q5_1_val2d.csv", index=False)


def fit_best_models():
    # Best settings from Q4.3/Q4.4.
    tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=25, random_state=42)
    lin = SVC(kernel="linear", C=0.5)
    rbf = SVC(kernel="rbf", C=1.0, gamma=0.1)
    for m in [tree, lin, rbf]:
        m.fit(X2_svm.to_numpy(), y_svm.to_numpy())
    return tree, lin, rbf


tree, lin, rbf = fit_best_models()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, model, name in [(axes[0], tree, "decision tree"),
                        (axes[1], lin, "linear SVM"),
                        (axes[2], rbf, "RBF SVM")]:
    decision_surface(ax, model, X2_val, y_val, name)
fig.suptitle(f"2D decision boundaries over the validation points "
             f"(PC1 {var.explained_variance_ratio[0]:.2f}, PC2 {var.explained_variance_ratio[1]:.2f})")
fig.tight_layout()
fig.savefig(OUT_DIR / "q5_1_boundaries.png", dpi=150)
print(f"\nsaved outputs -> {OUT_DIR}")