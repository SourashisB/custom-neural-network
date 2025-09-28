# tests/test_sequential.py

import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.model import Sequential


def test_sequential_forward_backward_shapes():
    N, D, H, C = 10, 4, 7, 3
    X = np.random.randn(N, D).astype("float32")
    model = Sequential([
        Dense(D, H, weight_init="xavier_uniform"),
        ReLU(),
        Dense(H, C, weight_init="he_normal"),
    ])

    out = model.forward(X, training=True)
    assert out.shape == (N, C)

    dUp = np.random.randn(*out.shape).astype("float32")
    dX = model.backward(dUp)
    assert dX.shape == (N, D)


def test_parameters_and_zero_grad():
    N, D, H = 5, 3, 4
    X = np.random.randn(N, D).astype("float32")
    model = Sequential([
        Dense(D, H),
        ReLU(),
        Dense(H, 2),
    ])

    out = model.forward(X, training=True)
    dUp = np.random.randn(*out.shape).astype("float32")
    model.backward(dUp)

    # Ensure parameters iterator returns grads and params
    params = list(model.parameters())
    assert len(params) == 4  # two dense layers: W,b for each
    for p in params:
        assert "param" in p and "grad" in p
        # grads should be same shape
        assert p["param"].shape == p["grad"].shape

    # After zero_grad, all grads become zero
    model.zero_grad()
    for p in model.parameters():
        assert np.allclose(p["grad"], 0.0)