"""Dataset loading, validation, and train/val splitting utilities for EstateIQ."""

import os
import csv
from typing import Tuple, Optional, Any, Dict, List
import numpy as np

from ml.common.constants import (
    TARGET_COLUMN,
    DEFAULT_TRAIN_CLEANED_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VAL_SIZE,
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_raw_dataset_records(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load CSV records using standard library csv.DictReader.

    Returns:
        header: list of column names
        records: list of row dicts
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for r in reader:
            records.append(dict(r))

    return list(header), records


def separate_features_and_target_records(
    records: List[Dict[str, str]],
    target_column: str = TARGET_COLUMN,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Separate target from feature records, guaranteeing no leakage.

    Raises:
        ValueError: If target_column is missing from records.
    """
    if not records:
        return [], []

    if target_column not in records[0]:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset records. "
            "Cannot perform supervised data preparation."
        )

    X_records = []
    y_values = []
    for r in records:
        r_copy = dict(r)
        target_val = float(r_copy.pop(target_column))
        X_records.append(r_copy)
        y_values.append(target_val)

    # Verification: Ensure target column is completely absent in every X record
    for r in X_records:
        if target_column in r:
            raise RuntimeError("CRITICAL: Target column leaked into feature record!")

    return X_records, y_values


def train_val_split_indices(
    n_samples: int,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[List[int], List[int]]:
    """Generate reproducible train and validation partition indices."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_val = int(n_samples * val_size)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    return train_indices, val_indices


def load_and_split_training_data(
    data_path: str = DEFAULT_TRAIN_CLEANED_PATH,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Any:
    """Load training data, extract target, and split into train and validation sets.

    When pandas is available, returns:
        (X_train, X_val, y_train, y_val) as (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
    Otherwise raises ImportError.
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for loading DataFrames. "
            "Please install pandas or use load_raw_dataset_records."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset file not found at: {data_path}")

    # Explicit ban on test.csv or sample_submission.csv
    base_name = os.path.basename(data_path).lower()
    if "test" in base_name or "sample_sub" in base_name:
        raise ValueError(
            f"Prohibited dataset '{data_path}' specified as training data! "
            "Test and sample submission files must NEVER be used for supervised training."
        )

    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' is missing from '{data_path}'."
        )

    # Separate target explicitly
    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    # Confirm target is absent from X
    assert TARGET_COLUMN not in X.columns, "Target column must not be present in X"

    train_idx, val_idx = train_val_split_indices(
        len(df), val_size=val_size, random_state=random_state
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    return X_train, X_val, y_train, y_val
