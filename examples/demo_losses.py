# examples/demo_losses.py

import numpy as np

from src.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MSELoss


def demo_mse():
    rng = np.random.default_rng(0)
    y_pred = rng.standard_normal((5, 3)).astype("float32")
    y_true = rng.standard_normal((5, 3)).astype("float32")
    mse = MSELoss()
    loss, grad = mse.forward_backward(y_pred, y_true)
    print("MSE loss:", loss, "grad shape:", grad.shape)


def demo_bce_logits():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((8, 1)).astype("float32")
    y = (rng.random((8, 1)) > 0.5).astype("float32")
    bce = BinaryCrossEntropy(from_logits=True)
    loss, grad = bce.forward_backward(logits, y)
    print("BCE (logits) loss:", loss, "grad mean:", float(grad.mean()))


def demo_bce_probs():
    rng = np.random.default_rng(2)
    p = rng.random((8, 1)).astype("float32")
    y = (rng.random((8, 1)) > 0.5).astype("float32")
    bce = BinaryCrossEntropy(from_logits=False)
    loss, grad = bce.forward_backward(p, y)
    print("BCE (probs) loss:", loss, "grad mean:", float(grad.mean()))


def demo_cce_logits():
    rng = np.random.default_rng(3)
    N, C = 6, 4
    logits = rng.standard_normal((N, C)).astype("float32")
    y_idx = rng.integers(0, C, size=N)
    cce = CategoricalCrossEntropy(from_logits=True)
    loss, grad = cce.forward_backward(logits, y_idx)
    print("CCE (logits) loss:", loss, "grad shape:", grad.shape, "row sum:", float(np.abs(grad.sum(axis=1)).mean()))


def demo_cce_probs_onehot():
    rng = np.random.default_rng(4)
    N, C = 5, 3
    p = rng.random((N, C)).astype("float32")
    p = p / p.sum(axis=1, keepdims=True)
    y_onehot = np.eye(C, dtype="float32")[rng.integers(0, C, size=N)]
    cce = CategoricalCrossEntropy(from_logits=False)
    loss, grad = cce.forward_backward(p, y_onehot)
    print("CCE (probs, onehot) loss:", loss, "grad norm:", float(np.linalg.norm(grad)))


if __name__ == "__main__":
    demo_mse()
    demo_bce_logits()
    demo_bce_probs()
    demo_cce_logits()
    demo_cce_probs_onehot()