# src/model.py

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


class Sequential:
    """
    Minimal Sequential container.

    Features:
      - add(layer): append a layer
      - forward(X, training=True): pass through layers
      - backward(dY): backpropagate through layers in reverse
      - parameters(): yield parameter dicts {"param": ndarray, "grad": ndarray}
      - zero_grad(): set all parameter grads to zero
      - __call__(X, training=True) -> forward convenience
    """

    def __init__(self, layers: Sequence[object] | None = None):
        self.layers: List[object] = []
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: object) -> None:
        # Expect each layer to implement forward, backward, parameters, zero_grad (if it has params)
        missing = [name for name in ("forward", "backward") if not hasattr(layer, name)]
        if missing:
            raise TypeError(f"Layer is missing required methods: {missing}")
        self.layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, dY: np.ndarray) -> np.ndarray:
        grad = dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> Iterable[Dict[str, np.ndarray]]:
        """
        Iterate over all trainable parameters across layers.
        Each parameter is a dict containing:
          - "param": np.ndarray (weights/biases)
          - "grad":  np.ndarray (same shape as param)
        """
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    yield p

    def zero_grad(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    # Convenience
    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(X, training=training)

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)