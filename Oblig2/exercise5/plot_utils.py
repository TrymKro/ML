"""Shared plotting helper for the 2D decision-boundary figures (Q5.x).

All Q5 experiments work on a two-column DataFrame (the PCA projection), so
the classifier is passed in on its own and the grid is built over the two
columns.
"""

import numpy as np
from matplotlib import pyplot as plt


def decision_surface(ax, model, X2, y2, title, show_sv=False):
    """Plot a decision boundary on ax for a model trained on 2D points."""
    lo = X2.min(axis=0).values
    hi = X2.max(axis=0).values
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    xx, yy = np.meshgrid(
        np.linspace(lo[0], hi[0], 300),
        np.linspace(lo[1], hi[1], 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid).reshape(xx.shape)
    else:
        Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.45)
    ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.5)

    colors = np.where(y2.to_numpy() == 1, "#c44e52", "#55a868")
    ax.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=colors, s=8, alpha=0.55, edgecolors="none")
    if show_sv:
        sv = model.support_vectors_  # only the RBF/linear SVC has these
        ax.scatter(sv[:, 0], sv[:, 1], facecolors="none", edgecolors="tab:orange", s=40, linewidths=1.0)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")