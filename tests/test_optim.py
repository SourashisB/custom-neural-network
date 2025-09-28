# tests/test_optim.py

import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.losses import MSELoss
from src.model import Sequential
from src.optim import Adam, SGD


def test_sgd_updates_reduce_loss():
    np.random.seed(0)
    N, D, H = 128, 5, 16
    X = np.random.randn(N, D).astype("float32")
    Wtrue = np.random.randn(D, 1).astype("float32")
    y = X @ Wtrue + 0.1 * np.random.randn(N, 1).astype("float32")

    model = Sequential([Dense(D, H), ReLU(), Dense(H, 1)])
    loss_fn = MSELoss()
    opt = SGD(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.0)

    # Before
    y_pred = model.forward(X, training=True)
    loss0, _ = loss_fn.forward_backward(y_pred, y, backward=False)

    # One step
    y_pred = model.forward(X, training=True)
    loss, grad = loss_fn.forward_backward(y_pred, y, backward=True)
    model.backward(grad)
    opt.step(model.parameters())
    model.zero_grad()

    y_pred2 = model.forward(X, training=False)
    loss1, _ = loss_fn.forward_backward(y_pred2, y, backward=False)

    assert loss1 <= loss0 + 1e-6  # should not increase (allow tiny num diff)


def test_adam_bias_correction_and_decay_shapes():
    np.random.seed(1)
    N, D, H = 64, 3, 8
    X = np.random.randn(N, D).astype("float32")
    y = np.random.randn(N, 1).astype("float32")

    model = Sequential([Dense(D, H), ReLU(), Dense(H, 1)])
    loss_fn = MSELoss()
    opt = Adam(lr=1e-2, weight_decay=1e-3)

    # Single step sanity
    y_pred = model.forward(X, training=True)
    loss, dypred = loss_fn.forward_backward(y_pred, y, backward=True)
    model.backward(dypred)
    params = list(model.parameters())
    old_params = [p["param"].copy() for p in params]
    opt.step(params)
    model.zero_grad()

    # Check params changed and shapes preserved
    for old, p in zip(old_params, params):
        assert old.shape == p["param"].shape
        assert not np.allclose(old, p["param"])