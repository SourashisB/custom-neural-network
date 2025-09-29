
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import normalize_minmax, train_val_split


Array = np.ndarray


@dataclass
class DatasetSplit:
    X_train: Array
    y_train: Array
    X_val: Array
    y_val: Array
    X_test: Optional[Array] = None
    y_test: Optional[Array] = None


def load_iris_pandas(
    csv_path: str,
    feature_cols: Optional[list[str]] = None,
    target_col: Optional[str] = None,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    normalize: bool = True,
) -> DatasetSplit:
    """
    Load Iris dataset from a local CSV using pandas.

    Assumptions:
      - If target_col is None, tries 'target' or 'species'.
      - If feature_cols is None, uses all numeric columns except target.

    Returns float32 features and int labels.
    """
    df = pd.read_csv(csv_path)
    # Infer target column
    if target_col is None:
        for cand in ["target", "species", "label", "class"]:
            if cand in df.columns:
                target_col = cand
                break
    if target_col is None:
        raise ValueError("Could not infer target column; please specify target_col.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].to_numpy(dtype="float32")
    y_raw = df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy()
    else:
        # factorize classes into 0..C-1
        y, _ = pd.factorize(y_raw)
    y = y.astype("int64")

    # Normalize features to [0,1] per column if requested
    if normalize:
        X = normalize_minmax(X, axis=0)

    # Shuffle
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(seed))
        np.random.shuffle(idx)
        if seed is not None:
            np.random.set_state(rng_state)
    X = X[idx]
    y = y[idx]

    # Split into train/val/test
    if not (0 <= test_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("val_ratio in (0,1) and test_ratio in [0,1) required.")
    if test_ratio > 0:
        split_test = int((1 - test_ratio) * N)
        X_rem, X_test = X[:split_test], X[split_test:]
        y_rem, y_test = y[:split_test], y[split_test:]
    else:
        X_rem, y_rem = X, y
        X_test, y_test = None, None

    X_train, y_train, X_val, y_val = train_val_split(X_rem, y_rem, val_ratio=val_ratio, seed=seed, shuffle=True)
    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)


def load_mnist_npz(
    npz_path: str,
    split: str = "train",  # "train" or "test"
    normalize: bool = True,
) -> Tuple[Array, Array]:
    """
    Load MNIST from a local NPZ file. Expected keys:
      - For common numpy mnist.npz: x_train, y_train, x_test, y_test
    Returns:
      X: (N, 784) float32 if grayscale 28x28, normalized to [0,1] if requested
      y: (N,) int64 labels
    """
    data = np.load(npz_path)
    if split == "train":
        X = data["x_train"]
        y = data["y_train"]
    elif split == "test":
        X = data["x_test"]
        y = data["y_test"]
    else:
        raise ValueError("split must be 'train' or 'test'")

    # Flatten if needed
    if X.ndim == 3:  # (N, H, W)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 4:  # (N, H, W, C)
        X = X.reshape(X.shape[0], -1)

    X = X.astype("float32")
    if normalize:
        # MNIST pixel range typically 0..255
        X /= 255.0
        X = np.clip(X, 0.0, 1.0)
    y = y.astype("int64")
    return X, y


def load_mnist_csv(
    csv_path: str,
    has_header: bool = True,
    label_first: bool = True,
    normalize: bool = True,
) -> Tuple[Array, Array]:
    """
    Load MNIST from a local CSV:
      - If label_first=True, assumes first column is label, rest are pixels.
      - Otherwise, last column is label.

    Returns:
      X: (N, 784) float32 normalized to [0,1] if requested
      y: (N,) int64 labels
    """
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    if label_first:
        y = df.iloc[:, 0].to_numpy().astype("int64")
        X = df.iloc[:, 1:].to_numpy().astype("float32")
    else:
        y = df.iloc[:, -1].to_numpy().astype("int64")
        X = df.iloc[:, :-1].to_numpy().astype("float32")

    if normalize:
        X /= 255.0
        X = np.clip(X, 0.0, 1.0)
    return X, y


def split_train_val_test(
    X: Array,
    y: Array,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> DatasetSplit:
    """
    Generic split into train/val/test with shuffling and seed.
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(int(seed))
        np.random.shuffle(idx)
        if seed is not None:
            np.random.set_state(rng_state)
    X = X[idx]
    y = y[idx]

    if not (0 <= test_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("val_ratio in (0,1) and test_ratio in [0,1) required.")
    n_test = int(np.floor(test_ratio * N))
    n_val = int(np.floor(val_ratio * (N - n_test)))
    n_train = N - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)