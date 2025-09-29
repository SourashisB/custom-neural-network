# tests/test_training.py

import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.losses import CategoricalCrossEntropy
from src.model import Sequential
from src.optim import SGD
from src.training import Trainer, TrainerConfig


def test_trainer_runs_and_improves_simple_task():
    np.random.seed(0)
    N, D, C = 300, 6, 3
    X = np.random.randn(N, D).astype("float32")
    W = np.random.randn(D, C).astype("float32")
    logits = X @ W + 0.3 * np.random.randn(N, C).astype("float32")
    y_idx = logits.argmax(axis=1)
    y = np.eye(C, dtype="float32")[y_idx]

    model = Sequential([Dense(D, 32, weight_init="he_normal"), ReLU(), Dense(32, C, weight_init="xavier_uniform")])
    loss = CategoricalCrossEntropy(from_logits=True)
    opt = SGD(lr=0.1, momentum=0.9, nesterov=True)

    cfg = TrainerConfig(epochs=5, batch_size=32, val_ratio=0.2, task="multiclass", seed=123)
    trainer = Trainer(model, opt, loss, cfg)

    # Before training
    logits0 = model.forward(X, training=False)
    loss0, _ = loss.forward_backward(logits0, y, backward=False)

    history = trainer.fit(X, y)
    logits1 = model.forward(X, training=False)
    loss1, _ = loss.forward_backward(logits1, y, backward=False)

    assert loss1 <= loss0 + 1e-5
    assert len(history["loss"]) >= 1