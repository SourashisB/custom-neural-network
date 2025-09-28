# src/initializers.py

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def zeros(shape: Tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    """
    Return an array of zeros with given shape and dtype.
    """
    return np.zeros(shape, dtype=dtype)


def xavier_uniform(
    shape: Tuple[int, ...],
    gain: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.

    For weight matrix of shape (fan_in, fan_out), samples from U(-a, a),
    where a = gain * sqrt(6 / (fan_in + fan_out)).

    Supports tensors; fans are computed from the first two dims when available.
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    fan_in, fan_out = _compute_fans(shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    arr = np.random.uniform(-limit, limit, size=shape).astype(dtype)
    if seed is not None:
        np.random.set_state(rng_state)
    return arr


def he_normal(
    shape: Tuple[int, ...],
    gain: float = 1.0,
    dtype: str = "float32",
    seed: int | None = None,
) -> np.ndarray:
    """
    He/Kaiming normal initialization.

    For weight matrix of shape (fan_in, fan_out), samples from N(0, std^2),
    where std = gain * sqrt(2 / fan_in). Suitable for ReLU-family activations.
    """
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    fan_in, _ = _compute_fans(shape)
    std = gain * math.sqrt(2.0 / fan_in) if fan_in > 0 else 1.0
    arr = (np.random.randn(*shape) * std).astype(dtype)
    if seed is not None:
        np.random.set_state(rng_state)
    return arr


def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out from tensor shape.
    - For Linear weights (fan_in, fan_out)
    - For Conv weights (out_channels, in_channels, k1, k2, ...)
    """
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        fan_in = shape[0]
        fan_out = shape[0]
        return fan_in, fan_out
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
        return fan_in, fan_out
    # Conv-like
    receptive_field_size = 1
    for s in shape[2:]:
        receptive_field_size *= s
    fan_in = shape[1] * receptive_field_size
    fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out