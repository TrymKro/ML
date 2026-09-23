"""Metric helpers used across the modelling exercises.

Returns a small dict of classification metrics; the class of interest is
always the positive (income > 50K) class.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, y_pred, y_score):
    """Accuracy, balanced accuracy, precision/recall/F1 and ROC-AUC."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def mean_std(values):
    """Format a mean +- std string."""
    return f"{np.mean(values):.4f} +- {np.std(values):.4f}"