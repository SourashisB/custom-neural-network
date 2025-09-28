# src/utils.py

from __future__ import annotations

import math
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """
    Set global RNG seed for NumPy and Python's hash seed for reproducibility.
    """
    np.random.seed(seed)
    # Python's hash seed influences hashing-based randomization in some libs
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)


def shuffle_in_unison(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Shuffle X (and y if provided) with the same random permutation.

    Params:
        X: array of shape (N, ...)
        y: optional array of shape (N, ...) or (N,)
        seed: optional seed for determinism

    Returns:
        X_shuffled, y_shuffled (y_shuffled is None if y is None)
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    N = X.shape[0]
    perm = np.random.permutation(N)
    Xs = X[perm]
    ys = y[perm] if y is not None else None
    if seed is not None:
        np.random.set_state(rng_state)
    return Xs, ys


def one_hot(
    y: Union[np.ndarray, Iterable[int]],
    num_classes: Optional[int] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Convert integer class indices to one-hot matrix.

    Params:
        y: shape (N,) int labels or iterable
        num_classes: optional; if None, inferred as max(y)+1
        dtype: output dtype

    Returns:
        Y: shape (N, C)
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"one_hot expects 1D array of class indices, got shape {y.shape}")
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size > 0 else 0
    Y = np.zeros((y.shape[0], num_classes), dtype=dtype)
    if y.size > 0 and num_classes > 0:
        Y[np.arange(y.shape[0]), y.astype(int)] = 1
    return Y


def batch_iterator(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
    """
    Yield mini-batches of (X, y).

    Params:
        X: shape (N, ...)
        y: optional labels with first dimension N
        batch_size: size of each batch
        shuffle: shuffle before iterating
        drop_last: if True, drop the last incomplete batch
        seed: optional seed for deterministic shuffling

    Yields:
        (xb, yb) where xb shape (B, ...) and yb shape (B, ...) or None
    """
    N = X.shape[0]
    if y is not None and y.shape[0] != N:
        raise ValueError(f"X and y must have the same first dimension. Got {N} and {y.shape[0]}")
    indices = np.arange(N)
    if shuffle:
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        np.random.shuffle(indices)
        if seed is not None:
            np.random.set_state(rng_state)

    for start in range(0, N, batch_size):
        end = start + batch_size
        if end > N and drop_last:
            break
        batch_idx = indices[start:end]
        xb = X[batch_idx]
        yb = y[batch_idx] if y is not None else None
        yield xb, yb


def train_val_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    val_ratio: float = 0.2,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Split arrays into train and validation sets.

    Params:
        X: shape (N, ...)
        y: optional shape (N, ...)
        val_ratio: fraction of data for validation (0 < val_ratio < 1)
        seed: optional seed for deterministic split
        shuffle: whether to shuffle before split

    Returns:
        X_train, y_train, X_val, y_val
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        np.random.shuffle(indices)
        if seed is not None:
            np.random.set_state(rng_state)
    split = int(math.floor((1.0 - val_ratio) * N))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    if y is None:
        return X_train, None, X_val, None
    y_train = y[train_idx]
    y_val = y[val_idx]
    return X_train, y_train, X_val, y_val


def to_numpy(x) -> np.ndarray:
    """
    Ensure input is a NumPy array (no-op if already np.ndarray).
    """
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def normalize_minmax(
    X: np.ndarray,
    axis: Optional[int] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Min-max normalize X to [0,1] along axis.

    If axis is None, uses global min/max.
    """
    X = np.asarray(X)
    if axis is None:
        xmin = X.min()
        xmax = X.max()
    else:
        xmin = X.min(axis=axis, keepdims=True)
        xmax = X.max(axis=axis, keepdims=True)
    return (X - xmin) / (xmax - xmin + eps)


def zscore(
    X: np.ndarray,
    axis: Optional[int] = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score standardization: (X - mean) / std
    """
    X = np.asarray(X)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + eps)


def ensure_2d(X: np.ndarray) -> np.ndarray:
    """
    Ensure X is 2D: if 1D becomes (N, 1), if 2D keep, else flatten last dims.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim == 2:
        return X
    N = X.shape[0]
    return X.reshape(N, -1)


def load_iris_dataframe() -> pd.DataFrame:
    """
    Load Iris dataset via pandas remote CSV alternative if available locally,
    or raise a helpful message. This function assumes you have a local iris CSV.
    Not using external URLs by design.

    Returns:
        DataFrame with features and species label column named 'target' if present.
    """
    # Placeholder to encourage user-provided path
    raise FileNotFoundError(
        "Please provide a local Iris CSV file and load it with pandas.read_csv. "
        "This scaffold avoids external URLs by design."
    )