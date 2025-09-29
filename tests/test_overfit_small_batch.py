# tests/test_overfit_small_batch.py

import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.losses import CategoricalCrossEntropy, MSELoss
from src.model import Sequential
from src.optim import Adam, SGD
from src.training import Trainer, TrainerConfig
from src.utils import one_hot, set_seed


def test_overfit_tiny_multiclass_batch():
    set_seed(123)
    N, D, C = 16, 8, 3
    X = np.random.randn(N, D).astype("float32")
    Wtrue = np.random.randn(D, C).astype("float32")
    logits = X @ Wtrue + 0.1 * np.random.randn(N, C).astype("float32")
    y_idx = logits.argmax(axis=1)
    y = one_hot(y_idx, C)

    model = Sequential([Dense(D, 32, weight_init="he_normal"), ReLU(), Dense(32, C, weight_init="xavier_uniform")])
    loss = CategoricalCrossEntropy(from_logits=True)
    opt = Adam(lr=0.05)

    cfg = TrainerConfig(epochs=200, batch_size=N, val_ratio=0.25, task="multiclass", seed=1, shuffle=True)
    trainer = Trainer(model, opt, loss, cfg)

    history = trainer.fit(X, y)
    final_acc = history["acc"][-1]
    assert final_acc > 0.95  # should easily overfit


def test_overfit_tiny_regression_batch():
    set_seed(321)
    N, D = 16, 6
    X = np.random.randn(N, D).astype("float32")
    Wtrue = np.random.randn(D, 1).astype("float32")
    y = X @ Wtrue + 0.05 * np.random.randn(N, 1).astype("float32")

    model = Sequential([Dense(D, 32), ReLU(), Dense(32, 1)])
    loss = MSELoss()
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)

    # Simple manual training to avoid classification metrics
    for _ in range(500):
        y_pred = model.forward(X, training=True)
        l, d = loss.forward_backward(y_pred, y, backward=True)
        model.backward(d)
        opt.step(model.parameters())
        model.zero_grad()

    y_pred = model.forward(X, training=False)
    final_mse, _ = loss.forward_backward(y_pred, y, backward=False)
    assert final_mse < 1e-3