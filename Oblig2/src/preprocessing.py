"""Leak-free preprocessing shared by the modelling exercises.

Implements the decisions from Exercise 1 as sklearn steps so they fit only on
the training folds: '?' -> 'Unknown' (Q1.1), rare categories -> 'other'
(Q1.3), one-hot encoding with the dummy variable dropped (Q1.4), and
standardizing the numeric features for the SVM branch only (Q1.7). Both
branches keep the raw numeric values (Q1.2).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# education is a pure copy of education-num (1:1 mapping, association 1.00 in
# Q1.5), so it is left out of the encoder and the numerics already include
# education-num.
CAT_ENCODE = [c for c in CATEGORICAL_COLUMNS if c != "education"]

# From Q1.3: keep categories seen in at least 162 rows, everything rarer
# becomes 'other'. An absolute row count, not a percentage, because the
# encoder is refitted on training folds smaller than the training set.
# 162 is 0.55% of the 29304-row training split and 0.33% of the merged file.
MIN_COUNT = 162


class RareGroup(BaseEstimator, TransformerMixin):
    """Fold categories seen in fewer than min_count rows into 'other'.

    The keep-set is learned in fit (the training fold), so a category can
    only survive because the training data contains it.
    """

    def __init__(self, min_count=MIN_COUNT):
        self.min_count = min_count

    def _as_df(self, X):
        # the ColumnTransformer hands us a numpy array; the categorical
        # columns arrive in the order defined above
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=CAT_ENCODE)

    def fit(self, X, y=None):
        X = self._as_df(X)
        self.keep_ = {}
        for col in X.columns:
            counts = X[col].value_counts()
            self.keep_[col] = counts[counts >= self.min_count].index
        return self

    def transform(self, X):
        X = self._as_df(X)
        out = X.copy()
        for col in X.columns:
            out[col] = X[col].where(X[col].isin(self.keep_[col]), "other")
        return out


def categorical_step():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("group", RareGroup(min_count=MIN_COUNT)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )


def tree_pipeline(**tree_kwargs):
    """Pipeline for tree models: numerics pass through untouched."""
    tree_kwargs.setdefault("random_state", 42)
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_COLUMNS),
            ("cat", categorical_step(), CAT_ENCODE),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("model", DecisionTreeClassifier(**tree_kwargs)),
        ]
    )


def svm_pipeline(svm_kwargs):
    """Pipeline for SVMs: the numerics are only standardised (Q1.2 keeps them)."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", categorical_step(), CAT_ENCODE),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("model", SVC(**svm_kwargs)),
        ]
    )