# tests/test_layers_and_activations.py

import numpy as np

from src.activations import ReLU, Sigmoid, Softmax, Tanh
from src.layers import Dense


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
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


def test_dense_forward_backward_gradcheck():
    rng = np.random.default_rng(0)
    N, D, H = 4, 3, 5
    X = rng.standard_normal((N, D)).astype("float64")  # double for gradcheck stability
    dense = Dense(D, H, weight_init="xavier_uniform", dtype="float64")
    Y = rng.standard_normal((N, H)).astype("float64")

    # Define loss: 0.5 * ||dense(X) - Y||^2
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
    dOut = diff  # dL/dOut for MSE

    dense.backward(dOut)  # populates dW, db

    num_dW = numerical_gradient(lambda W: loss_W(W), dense.W.copy())
    num_db = numerical_gradient(lambda b: loss_b(b), dense.b.copy())

    rel_dW = np.linalg.norm(dense.dW - num_dW) / (np.linalg.norm(dense.dW) + np.linalg.norm(num_dW) + 1e-12)
    rel_db = np.linalg.norm(dense.db - num_db) / (np.linalg.norm(dense.db) + np.linalg.norm(num_db) + 1e-12)

    assert rel_dW < 1e-6
    assert rel_db < 1e-6


def test_relu_backward_mask():
    x = np.array([[-1.0, 0.0, 2.0]], dtype="float32")
    relu = ReLU()
    y = relu.forward(x)
    dy = np.ones_like(x)
    dx = relu.backward(dy)
    assert np.allclose(y, [[0.0, 0.0, 2.0]])
    assert np.allclose(dx, [[0.0, 0.0, 1.0]])


def test_sigmoid_tanh_shapes():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 4)).astype("float32")
    sig = Sigmoid()
    tanh = Tanh()
    y1 = sig.forward(x)
    y2 = tanh.forward(x)
    dx1 = sig.backward(np.ones_like(y1))
    dx2 = tanh.backward(np.ones_like(y2))
    assert y1.shape == x.shape and y2.shape == x.shape
    assert dx1.shape == x.shape and dx2.shape == x.shape


def test_softmax_stable_rowsum_and_backward():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 5)).astype("float32") * 10.0  # large logits to test stability
    sm = Softmax(axis=1)
    y = sm.forward(x)
    rowsums = y.sum(axis=1)
    assert np.allclose(rowsums, np.ones_like(rowsums), atol=1e-6)

    dy = rng.standard_normal(y.shape).astype("float32")
    dx = sm.backward(dy)
    assert dx.shape == y.shape