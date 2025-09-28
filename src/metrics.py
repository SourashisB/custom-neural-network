# src/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Array = np.ndarray


def accuracy(
    y_pred: Array,
    y_true: Array,
    task: Literal["multiclass", "binary"] = "multiclass",
    threshold: float = 0.5,
) -> float:
    """
    Compute accuracy for classification.

    - multiclass:
        y_pred: (N, C) scores or probabilities
        y_true: (N,) class indices or (N, C) one-hot
    - binary:
        y_pred: (N, 1) logits/probs or (N,) values
        y_true: (N, 1) or (N,) binary labels in {0,1}

    Returns:
        float accuracy in [0,1]
    """
    y_true = np.asarray(y_true)

    if task == "multiclass":
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be 2D for multiclass. Got {y_pred.shape}")
        preds = y_pred.argmax(axis=1)
        if y_true.ndim == 2:
            y_true_idx = y_true.argmax(axis=1)
        else:
            y_true_idx = y_true.astype(int)
        return float((preds == y_true_idx).mean())

    # binary
    yp = np.asarray(y_pred).squeeze()
    yt = y_true.squeeze().astype(int)
    # If shape is (N,), works; if (N,1), squeeze handles it.
    if yp.ndim != 1:
        raise ValueError(f"Binary y_pred must be 1D after squeeze. Got {yp.shape}")
    # If values look like logits or probs, threshold applies the same to probs/logits if user passes probs
    # For logits, caller should pass sigmoid(logits) here if needed; to keep generic we treat as probs.
    preds = (yp >= threshold).astype(int)
    return float((preds == yt).mean())


def confusion_matrix(
    y_pred: Array,
    y_true: Array,
    num_classes: Optional[int] = None,
) -> Array:
    """
    Multiclass confusion matrix.

    y_pred: (N, C) scores/probs
    y_true: (N,) indices or (N, C) one-hot

    Returns:
        (C, C) matrix where rows=true class, cols=predicted class.
    """
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be (N, C), got {y_pred.shape}")
    N, C = y_pred.shape
    if num_classes is None:
        num_classes = C
    preds = y_pred.argmax(axis=1)
    yt = np.asarray(y_true)
    if yt.ndim == 2:
        yt = yt.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt, preds):
        cm[int(t), int(p)] += 1
    return cm