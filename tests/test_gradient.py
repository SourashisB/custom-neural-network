# tests/test_gradients.py

import numpy as np

from src.activations import ReLU, Sigmoid, Softmax, Tanh
from src.layers import Dense
from src.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MSELoss


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f(x)
        x[idx] = orig - eps
        f2 = f(x)
        x[idx] = orig
        grad[idx] = (f1 - f2) / (2 * eps)
        it.iternext()
    return grad


def test_dense_gradients_vs_numerical():
    rng = np.random.default_rng(0)
    N, D, H = 4, 3, 5
    X = rng.standard_normal((N, D)).astype("float64")
    dense = Dense(D, H, weight_init="xavier_uniform", dtype="float64")
    Y = rng.standard_normal((N, H)).astype("float64")

    def loss_W(W):
        out = X @ W + dense.b
        diff = out - Y
        return 0.5 * np.sum(diff * diff)

    def loss_b(b):
        out = X @ dense.W + b
        diff = out - Y
        return 0.5 * np.sum(diff * diff)

    out = dense.forward(X)
    diff = out - Y
    dOut = diff
    dense.backward(dOut)

    num_dW = numerical_gradient(lambda W: loss_W(W), dense.W.copy())
    num_db = numerical_gradient(lambda b: loss_b(b), dense.b.copy())

    rel_dW = np.linalg.norm(dense.dW - num_dW) / (np.linalg.norm(dense.dW) + np.linalg.norm(num_dW) + 1e-12)
    rel_db = np.linalg.norm(dense.db - num_db) / (np.linalg.norm(dense.db) + np.linalg.norm(num_db) + 1e-12)
    assert rel_dW < 1e-6
    assert rel_db < 1e-6


def test_activation_backward_correctness():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 7)).astype("float64")
    dUp = rng.standard_normal((6, 7)).astype("float64")

    # ReLU
    relu = ReLU()
    Y = relu.forward(X)
    dX = relu.backward(dUp)
    assert np.all((X <= 0) == (dX == 0))

    # Sigmoid numeric check
    sig = Sigmoid()
    S = sig.forward(X)
    dL_dS = dUp
    dX_sig = sig.backward(dL_dS)

    def f_sig(x):
        return np.sum(sig.forward(x) * dL_dS)

    num_dX_sig = numerical_gradient(f_sig, X.copy())
    rel = np.linalg.norm(dX_sig - num_dX_sig) / (np.linalg.norm(dX_sig) + np.linalg.norm(num_dX_sig) + 1e-12)
    assert rel < 1e-5

    # Tanh numeric check
    tanh = Tanh()
    T = tanh.forward(X)
    dX_tanh = tanh.backward(dL_dS)

    def f_tanh(x):
        return np.sum(tanh.forward(x) * dL_dS)

    num_dX_tanh = numerical_gradient(f_tanh, X.copy())
    rel2 = np.linalg.norm(dX_tanh - num_dX_tanh) / (np.linalg.norm(dX_tanh) + np.linalg.norm(num_dX_tanh) + 1e-12)
    assert rel2 < 1e-5

    # Softmax backward identity check: rowsum zero when composed with arbitrary dUp
    sm = Softmax(axis=1)
    P = sm.forward(X)
    dX_sm = sm.backward(dUp)
    rowsum = dX_sm.sum(axis=1)
    assert np.allclose(rowsum, 0.0, atol=1e-8)


def test_loss_gradients():
    rng = np.random.default_rng(2)

    # MSE
    mse = MSELoss()
    y_pred = rng.standard_normal((5, 3)).astype("float64")
    y_true = rng.standard_normal((5, 3)).astype("float64")

    def f_mse(x):
        loss, _ = mse.forward_backward(x, y_true, backward=False)
        return loss

    _, grad_mse = mse.forward_backward(y_pred, y_true, backward=True)
    num_grad_mse = numerical_gradient(f_mse, y_pred.copy())
    rel_mse = np.linalg.norm(grad_mse - num_grad_mse) / (np.linalg.norm(grad_mse) + np.linalg.norm(num_grad_mse) + 1e-12)
    assert rel_mse < 1e-6

    # BCE (logits)
    bce = BinaryCrossEntropy(from_logits=True)
    z = rng.standard_normal((7, 1)).astype("float64")
    y = (rng.random((7, 1)) > 0.5).astype("float64")

    def f_bce(x):
        loss, _ = bce.forward_backward(x, y, backward=False)
        return loss

    _, grad_bce = bce.forward_backward(z, y, backward=True)
    num_grad_bce = numerical_gradient(f_bce, z.copy())
    rel_bce = np.linalg.norm(grad_bce - num_grad_bce) / (np.linalg.norm(grad_bce) + np.linalg.norm(num_grad_bce) + 1e-10)
    assert rel_bce < 5e-4  # looser due to exp/log numerics

    # CCE (logits)
    cce = CategoricalCrossEntropy(from_logits=True)
    N, C = 6, 4
    z = rng.standard_normal((N, C)).astype("float64")
    y_idx = rng.integers(0, C, size=N)

    def f_cce(x):
        loss, _ = cce.forward_backward(x, y_idx, backward=False)
        return loss

    _, grad_cce = cce.forward_backward(z, y_idx, backward=True)
    num_grad_cce = numerical_gradient(f_cce, z.copy())
    rel_cce = np.linalg.norm(grad_cce - num_grad_cce) / (np.linalg.norm(grad_cce) + np.linalg.norm(num_grad_cce) + 1e-10)
    assert rel_cce < 1e-4