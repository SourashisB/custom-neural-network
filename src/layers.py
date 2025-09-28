# src/layers.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .initializers import he_normal, xavier_uniform, zeros
from .tensor import assert_shape


class Layer:
    """
    Base class for layers. Each layer that has parameters should expose:
      - parameters(): list of dicts with "param" and "grad" numpy arrays
      - zero_grad(): set grads to zero
    """

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dY: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> List[Dict[str, np.ndarray]]:
        return []

    def zero_grad(self) -> None:
        pass


@dataclass
class Dense(Layer):
    in_features: int
    out_features: int
    weight_init: str = "xavier_uniform"
    bias_init: str = "zeros"
    dtype: str = "float32"

    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    dW: np.ndarray = field(init=False)
    db: np.ndarray = field(init=False)
    _cache_X: np.ndarray = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.weight_init == "xavier_uniform":
            self.W = xavier_uniform((self.in_features, self.out_features), dtype=self.dtype)
        elif self.weight_init == "he_normal":
            self.W = he_normal((self.in_features, self.out_features), dtype=self.dtype)
        else:
            raise ValueError(f"Unknown weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            self.b = zeros((self.out_features,), dtype=self.dtype)
        else:
            raise ValueError(f"Unknown bias_init: {self.bias_init}")

        self.dW = zeros(self.W.shape, dtype=self.dtype)
        self.db = zeros(self.b.shape, dtype=self.dtype)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        # X: (N, D), W: (D, H), b: (H,)
        assert_shape(X, (None, self.in_features), name="Dense.forward.X")
        self._cache_X = X
        out = X @ self.W + self.b  # (N, H)
        return out

    def backward(self, dY: np.ndarray) -> np.ndarray:
        # dY: (N, H)
        X = self._cache_X
        if X is None:
            raise RuntimeError("Dense.backward called before forward.")
        assert_shape(dY, (None, self.out_features), name="Dense.backward.dY")

        # Gradients
        # dW = X^T @ dY, db = sum over batch, dX = dY @ W^T
        self.dW[...] = X.T @ dY  # (D, N) @ (N, H) -> (D, H)
        self.db[...] = dY.sum(axis=0)  # (H,)
        dX = dY @ self.W.T  # (N, H) @ (H, D) -> (N, D)
        return dX

    def parameters(self) -> List[Dict[str, np.ndarray]]:
        return [
            {"param": self.W, "grad": self.dW},
            {"param": self.b, "grad": self.db},
        ]

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)