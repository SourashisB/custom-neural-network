
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def assert_shape(x: np.ndarray, expected: Sequence[Optional[int]], name: str = "tensor") -> None:
    """
    Assert that x has shape compatible with `expected`.

    Use None as wildcard for a dimension.

    Example:
        assert_shape(X, (None, D)) ensures X.ndim == 2 and X.shape[1] == D.
    """
    if x.ndim != len(expected):
        raise ValueError(f"{name} ndim mismatch: got {x.ndim}, expected {len(expected)}")
    for i, (got, exp) in enumerate(zip(x.shape, expected)):
        if exp is not None and got != exp:
            raise ValueError(f"{name} shape mismatch at dim {i}: got {got}, expected {exp}")


def to_one_hot(
    y: np.ndarray | Iterable[int],
    num_classes: Optional[int] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """
    One-hot like utils.one_hot; kept here for ergonomics close to tensor ops.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"to_one_hot expects 1D array of class indices, got shape {y.shape}")
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size > 0 else 0
    out = np.zeros((y.shape[0], num_classes), dtype=dtype)
    if y.size > 0 and num_classes > 0:
        out[np.arange(y.shape[0]), y.astype(int)] = 1
    return out