"""Shared constants and helpers used across the exercise scripts.

Centralising the column definition and the loader here guarantees every
script looks at exactly the same data, with the same handling of the '?'
missing-value marker used by the UCI Adult files.
"""

from pathlib import Path

import pandas as pd

# Directory holding the raw UCI files (adult.data, adult.test, ...).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The 14 attributes plus the target, as documented in adult.names.
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Numeric features (continuous) per adult.names.
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Categorical features (discrete) per adult.names.
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# The target column.
TARGET = "income"


def load_raw(train_file="adult.data", test_file="adult.test"):
    """Load the raw UCI files and return (train, test) DataFrames.

    The '?' marker is parsed as NaN and the annoying header line plus the
    trailing '.' on the test target values are handled here.
    """
    train = pd.read_csv(
        DATA_DIR / train_file,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )
    test = pd.read_csv(
        DATA_DIR / test_file,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,  # adult.test starts with the '|1x3 Cross validator' line
    )
    # The test target values carry a trailing '.', e.g. '<=50K.', so strip it.
    test[TARGET] = test[TARGET].str.rstrip(".")
    return train, test