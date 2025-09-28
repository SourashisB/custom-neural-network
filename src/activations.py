from __future__ import annotations

from typing import Optional

import numpy as np

from .tensor import assert_shape


class Activation:
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dY: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """
    ReLU with mask caching:
      forward: Y = max(0, X)
      backward: dX = dY * (X > 0)
    """

    def __init__(self):
        self._mask: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self._mask = X > 0
        return X * self._mask

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward.")
        assert dY.shape == self._mask.shape, "Shape mismatch in ReLU backward"
        return dY * self._mask


class Sigmoid(Activation):
    """
    Sigmoid with output caching:
      forward: Y = 1 / (1 + exp(-X))
      backward: dX = dY * Y * (1 - Y)
    Uses stable computations for large |X|.
    """

    def __init__(self):
        self._Y: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        # Stable sigmoid: handle large negative and positive values
        out = np.empty_like(X)
        pos_mask = X >= 0
        neg_mask = ~pos_mask

        out[pos_mask] = 1.0 / (1.0 + np.exp(-X[pos_mask]))
        exp_x = np.exp(X[neg_mask])
        out[neg_mask] = exp_x / (1.0 + exp_x)

        self._Y = out
        return out

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self._Y is None:
            raise RuntimeError("Sigmoid.backward called before forward.")
        return dY * self._Y * (1.0 - self._Y)


class Tanh(Activation):
    """
    Tanh with output caching:
      forward: Y = tanh(X)
      backward: dX = dY * (1 - Y^2)
    """

    def __init__(self):
        self._Y: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        Y = np.tanh(X)
        self._Y = Y
        return Y

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self._Y is None:
            raise RuntimeError("Tanh.backward called before forward.")
        return dY * (1.0 - self._Y**2)


class Softmax(Activation):
    """
    Stable softmax with log-sum-exp trick.

    forward:
      For each row i:
        z = X[i] - max(X[i])
        exp_z = exp(z)
        Y[i] = exp_z / sum(exp_z)

    backward:
      Given upstream gradient dY (dL/dY), general Jacobian-vector product:
        dX = Y * (dY - sum(dY * Y, axis=1, keepdims=True))

      Note:
      - When used with Categorical Cross-Entropy with logits, prefer a fused
        loss that directly computes dX = (softmax - one_hot) / N for stability.
        In that case you typically do NOT include a Softmax layer in the model.
    """

    def __init__(self, axis: int = 1):
        self.axis = axis
        self._Y: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        # Move softmax axis to last for stable operations, then move back
        if self.axis < 0:
            axis = X.ndim + self.axis
        else:
            axis = self.axis

        # For common case of 2D (N, C) with axis=1, simplify:
        if X.ndim == 2 and axis == 1:
            X_shift = X - X.max(axis=1, keepdims=True)
            exp_x = np.exp(X_shift)
            Y = exp_x / exp_x.sum(axis=1, keepdims=True)
            self._Y = Y
            return Y

        # Generic case
        X_shift = X - np.max(X, axis=axis, keepdims=True)
        exp_x = np.exp(X_shift)
        denom = np.sum(exp_x, axis=axis, keepdims=True)
        Y = exp_x / denom
        self._Y = Y
        return Y

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self._Y is None:
            raise RuntimeError("Softmax.backward called before forward.")
        Y = self._Y

        # For 2D case with axis=1: efficient formula
        if Y.ndim == 2 and self.axis in (1, -1):
            dot = (dY * Y).sum(axis=1, keepdims=True)  # (N, 1)
            dX = Y * (dY - dot)
            return dX

        # Generic case
        dot = np.sum(dY * Y, axis=self.axis, keepdims=True)
        dX = Y * (dY - dot)
        return dX