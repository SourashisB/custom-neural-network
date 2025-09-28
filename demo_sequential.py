# examples/demo_sequential.py

import numpy as np

from src.activations import ReLU, Softmax
from src.layers import Dense
from src.losses import CategoricalCrossEntropy, MSELoss
from src.model import Sequential
from src.utils import one_hot, set_seed, train_val_split


def demo_classification():
    set_seed(42)
    N, D, C = 128, 10, 3
    X = np.random.randn(N, D).astype("float32")
    y_idx = np.random.randint(0, C, size=N)
    y = one_hot(y_idx, C)

    model = Sequential([
        Dense(D, 32, weight_init="he_normal"),
        ReLU(),
        Dense(32, C, weight_init="xavier_uniform"),
        # For stable training with CE, omit Softmax and use from_logits=True
        # Softmax(),  # Uncomment if you prefer CE with probabilities
    ])

    loss_fn = CategoricalCrossEntropy(from_logits=True)

    # Simple manual one-epoch SGD step with fixed lr, to demonstrate API
    lr = 1e-1
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=7)

    # Forward
    logits = model.forward(X_tr, training=True)
    loss, dlogits = loss_fn.forward_backward(logits, y_tr, backward=True)

    # Backward
    model.backward(dlogits)

    # Naive SGD update
    for p in model.parameters():
        p["param"][...] -= lr * p["grad"]

    # Zero grads for next step
    model.zero_grad()

    # Evaluate
    logits_val = model.forward(X_val, training=False)
    val_loss, _ = loss_fn.forward_backward(logits_val, y_val, backward=False)
    preds = logits_val.argmax(axis=1)
    acc = (preds == y_val.argmax(axis=1)).mean()

    print("Train loss:", float(loss))
    print("Val loss:", float(val_loss), "Val acc:", float(acc))


def demo_regression():
    set_seed(0)
    N, D, H = 64, 5, 16
    X = np.random.randn(N, D).astype("float32")
    true_W = np.random.randn(D, 1).astype("float32")
    y = X @ true_W + 0.1 * np.random.randn(N, 1).astype("float32")

    model = Sequential([
        Dense(D, H, weight_init="xavier_uniform"),
        ReLU(),
        Dense(H, 1, weight_init="xavier_uniform"),
    ])
    mse = MSELoss()
    lr = 1e-2

    # One training step
    y_pred = model.forward(X, training=True)
    loss, dypred = mse.forward_backward(y_pred, y, backward=True)
    model.backward(dypred)
    for p in model.parameters():
        p["param"][...] -= lr * p["grad"]
    model.zero_grad()

    print("Regression step MSE:", float(loss))


if __name__ == "__main__":
    demo_classification()
    demo_regression()