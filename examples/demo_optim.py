# examples/demo_optim.py

import numpy as np

from src.activations import ReLU
from src.layers import Dense
from src.losses import CategoricalCrossEntropy, MSELoss
from src.model import Sequential
from src.optim import Adam, SGD
from src.utils import one_hot, set_seed, train_val_split


def demo_sgd_nesterov():
    set_seed(42)
    N, D, C = 128, 20, 4
    X = np.random.randn(N, D).astype("float32")
    y_idx = np.random.randint(0, C, size=N)
    y = one_hot(y_idx, C)

    model = Sequential([
        Dense(D, 64, weight_init="he_normal"),
        ReLU(),
        Dense(64, C, weight_init="xavier_uniform"),
    ])
    loss_fn = CategoricalCrossEntropy(from_logits=True)
    opt = SGD(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)

    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=7)

    for epoch in range(5):
        # Forward
        logits = model.forward(X_tr, training=True)
        loss, dlogits = loss_fn.forward_backward(logits, y_tr, backward=True)
        # Backward
        model.backward(dlogits)
        # Step
        opt.step(model.parameters())
        # Zero grads
        model.zero_grad()

        # Eval
        logits_val = model.forward(X_val, training=False)
        val_loss, _ = loss_fn.forward_backward(logits_val, y_val, backward=False)
        preds = logits_val.argmax(axis=1)
        acc = (preds == y_val.argmax(axis=1)).mean()
        print(f"[SGD+NAG] Epoch {epoch+1} | loss={loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.3f}")


def demo_adam_wd():
    set_seed(0)
    N, D, H = 200, 10, 32
    X = np.random.randn(N, D).astype("float32")
    true_W = np.random.randn(D, 1).astype("float32")
    y = X @ true_W + 0.5 * np.random.randn(N, 1).astype("float32")

    model = Sequential([
        Dense(D, H, weight_init="xavier_uniform"),
        ReLU(),
        Dense(H, 1, weight_init="xavier_uniform"),
    ])
    mse = MSELoss()
    opt = Adam(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    for step in range(20):
        y_pred = model.forward(X, training=True)
        loss, dypred = mse.forward_backward(y_pred, y, backward=True)
        model.backward(dypred)
        opt.step(model.parameters())
        model.zero_grad()
        if (step + 1) % 5 == 0:
            print(f"[Adam] Step {step+1} | MSE={loss:.4f}")

if __name__ == "__main__":
    demo_sgd_nesterov()
    demo_adam_wd()