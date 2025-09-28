# src/losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

Array = np.ndarray


def _ensure_2d_probs(y_pred: Array) -> Array:
    if y_pred.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (N, C). Got {y_pred.shape}")
    return y_pred


def _as_one_hot_or_indices(y_true: Array, num_classes: Optional[int] = None) -> Tuple[Array, bool]:
    """
    Normalize targets to either:
      - (N, C) one-hot matrix and flag True
      - (N,) class indices and flag False
    """
    y_true = np.asarray(y_true)
    if y_true.ndim == 2:
        # One-hot-like
        return y_true, True
    if y_true.ndim == 1:
        if num_classes is not None and y_true.max() >= num_classes:
            raise ValueError("y_true contains class index >= num_classes.")
        return y_true.astype(int), False
    raise ValueError(f"y_true must be shape (N,) or (N, C). Got {y_true.shape}")


def _safe_mean(x: Array) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


@dataclass
class MSELoss:
    """
    Mean Squared Error:
      L = mean((y_pred - y_true)^2)

    forward_backward(y_pred, y_true, backward=True) ->
      returns (loss_scalar, dL/dy_pred if backward else None)
    """

    reduction: str = "mean"  # currently only mean is supported

    def forward_backward(self, y_pred: Array, y_true: Array, backward: bool = True):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"MSELoss: y_pred and y_true must have same shape. {y_pred.shape} vs {y_true.shape}")
        diff = y_pred - y_true
        loss = _safe_mean(diff * diff)
        if not backward:
            return loss, None
        # d/dy (mean(diff^2)) = 2*diff / N_elements
        denom = np.prod(y_pred.shape, dtype=np.float64)
        grad = (2.0 / denom) * diff
        return loss, grad


@dataclass
class BinaryCrossEntropy:
    """
    Binary cross-entropy for binary classification per example:
      L = -mean( y*log(p) + (1-y)*log(1-p) )

    Stable implementation: compute using logits if provided, or probabilities with clamping.
    You can pass:
      - logits: set from_logits=True (preferred)
      - probabilities in (0,1): set from_logits=False

    Targets y_true can be 0/1 floats with the same shape as y_pred.

    If from_logits:
      L = mean( max(z,0) - z*y + log(1 + exp(-|z|)) )
      grad wrt logits: sigmoid(z) - y, then average over N
    """

    from_logits: bool = True
    eps: float = 1e-12

    def _sigmoid(self, z: Array) -> Array:
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def forward_backward(self, y_pred: Array, y_true: Array, backward: bool = True):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"BinaryCrossEntropy: y_pred and y_true must have same shape. {y_pred.shape} vs {y_true.shape}")
        N = y_pred.shape[0] if y_pred.ndim >= 1 else y_pred.size
        N = max(N, 1)

        if self.from_logits:
            z = y_pred
            # Loss: mean(max(z,0) - z*y + log(1 + exp(-|z|)))
            abs_z = np.abs(z)
            loss_vec = np.maximum(z, 0) - z * y_true + np.log1p(np.exp(-abs_z))
            loss = _safe_mean(loss_vec)
            if not backward:
                return loss, None
            # dL/dz = sigmoid(z) - y; then divide by N elements per-sample. For mean over batch, divide by N
            grad = (self._sigmoid(z) - y_true) / N
            return loss, grad.astype(y_pred.dtype)
        else:
            # y_pred are probabilities; clamp for stability
            p = np.clip(y_pred, self.eps, 1.0 - self.eps)
            loss_vec = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            loss = _safe_mean(loss_vec)
            if not backward:
                return loss, None
            grad = (p - y_true) / (p * (1.0 - p) + self.eps)
            grad = grad / N
            return loss, grad.astype(y_pred.dtype)


@dataclass
class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy for multi-class classification.

    Usage:
      - from_logits=True (preferred): pass raw scores (logits) of shape (N, C).
        Stable computation via log-softmax:
          loss_i = -log_softmax(z_i)[y_i]
        Grad:
          dL/dz = (softmax(z) - one_hot(y)) / N

      - from_logits=False: pass probabilities of shape (N, C).
        Loss:
          loss_i = -sum_j y_ij * log(p_ij)
        Grad:
          dL/dp = -(y / p) / N

    Targets:
      - y_true shape (N,) with class indices
      - or y_true shape (N, C) one-hot

    By default uses mean reduction across batch.
    """

    from_logits: bool = True
    eps: float = 1e-12

    def _log_softmax(self, z: Array) -> Array:
        z = _ensure_2d_probs(z)
        z_shift = z - z.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(z_shift).sum(axis=1, keepdims=True))
        return z_shift - logsumexp  # shape (N, C)

    def _softmax(self, z: Array) -> Array:
        z = _ensure_2d_probs(z)
        z_shift = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward_backward(
        self,
        y_pred: Array,
        y_true: Union[Array, np.ndarray],
        backward: bool = True,
    ):
        z_or_p = np.asarray(y_pred)
        if z_or_p.ndim != 2:
            raise ValueError(f"CategoricalCrossEntropy expects (N, C). Got {z_or_p.shape}")
        N, C = z_or_p.shape
        y_t, is_one_hot = _as_one_hot_or_indices(np.asarray(y_true), num_classes=C)

        if self.from_logits:
            # Loss via log-softmax
            log_probs = self._log_softmax(z_or_p)  # (N, C)
            if is_one_hot:
                # Sum over classes
                loss_vec = -np.sum(y_t * log_probs, axis=1)
            else:
                # Index per row
                loss_vec = -log_probs[np.arange(N), y_t]
            loss = _safe_mean(loss_vec)
            if not backward:
                return loss, None
            # Grad wrt logits: softmax - one_hot
            probs = np.exp(log_probs)  # softmax
            if is_one_hot:
                grad = (probs - y_t) / N
            else:
                grad = probs
                grad[np.arange(N), y_t] -= 1.0
                grad /= N
            return loss, grad.astype(z_or_p.dtype)
        else:
            # y_pred are probabilities
            p = np.clip(z_or_p, self.eps, 1.0)  # ensure positive
            if is_one_hot:
                loss_vec = -np.sum(y_t * np.log(p + self.eps), axis=1)
            else:
                loss_vec = -np.log(p[np.arange(N), y_t] + self.eps)
            loss = _safe_mean(loss_vec)
            if not backward:
                return loss, None
            if is_one_hot:
                grad = -(y_t / (p + self.eps)) / N
            else:
                grad = np.zeros_like(p)
                grad[np.arange(N), y_t] = -1.0 / (p[np.arange(N), y_t] + self.eps)
                grad /= N
            return loss, grad.astype(z_or_p.dtype)