"""Train/validation/test split shared by the modelling exercises.

The UCI files are concatenated into one frame, then split 60/20/20 with a
stratified split (class ratio preserved in all three sets), as required by
the assignment. The same folds are reused across exercises so every model is
compared on exactly the same training/validation data.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# make the code/ folder importable when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import TARGET, load_raw

RANDOM_STATE = 42


def make_split():
    """Return (X_train, X_val, X_test, y_train, y_val, y_test) plus the
    5 stratified folds of the training set, and the SVM subsample with its
    own folds."""
    train, test = load_raw()
    data = pd.concat([train, test], ignore_index=True)
    y = (data[TARGET] == ">50K").astype(int)
    X = data.drop(columns=[TARGET])

    # 20% test, then 25% of the remaining 80% -> validation, so 60/20/20.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_train, y_train))

    # The same folds, but on a stratified subsample of the training set,
    # so the SVM experiments (which scale badly with n) stay comparable.
    sample = train_test_split(
        X_train, y_train, test_size=len(X_train) - 5000,
        stratify=y_train, random_state=RANDOM_STATE,
    )
    X_svm, X_drop, y_svm, y_drop = sample
    svm_folds = list(cv.split(X_svm, y_svm))

    return X_train, X_val, X_test, y_train, y_val, y_test, folds, X_svm, y_svm, svm_folds


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, folds, *_ = make_split()
    print(f"train: {X_train.shape[0]} rows ({y_train.mean():.3f} positive)")
    print(f"val:   {X_val.shape[0]} rows ({y_val.mean():.3f} positive)")
    print(f"test:  {X_test.shape[0]} rows ({y_test.mean():.3f} positive)")
    print(f"folds: {len(folds)}")