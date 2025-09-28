# examples/demo_layers.py

import numpy as np

from src.activations import ReLU, Sigmoid, Softmax, Tanh
from src.layers import Dense
from src.tensor import to_one_hot


def demo_dense_and_activations():
    rng = np.random.default_rng(0)
    N, D, H, C = 8, 5, 10, 3

    X = rng.standard_normal((N, D)).astype("float32")

    # Dense layer with He init (good for ReLU)
    dense = Dense(D, H, weight_init="he_normal")
    relu = ReLU()
    out1 = dense.forward(X, training=True)
    out2 = relu.forward(out1, training=True)

    # Backward through ReLU -> Dense with random upstream gradient
    dUp = rng.standard_normal(out2.shape).astype("float32")
    dRelu = relu.backward(dUp)
    dX = dense.backward(dRelu)

    print("Dense forward shape:", out1.shape)
    print("ReLU forward shape:", out2.shape)
    print("dX shape:", dX.shape)
    print("dW norm:", float(np.linalg.norm(dense.dW)))
    print("db norm:", float(np.linalg.norm(dense.db)))

    # Test Sigmoid/Tanh cache
    sig = Sigmoid()
    tanh = Tanh()
    s = sig.forward(out1)
    t = tanh.forward(out1)
    ds = sig.backward(np.ones_like(s))
    dt = tanh.backward(np.ones_like(t))
    print("Sigmoid/Tanh backward mean magnitudes:", float(ds.mean()), float(dt.mean()))

    # Softmax forward/backward demo
    logits = rng.standard_normal((N, C)).astype("float32")
    sm = Softmax(axis=1)
    probs = sm.forward(logits)
    y = to_one_hot(np.arange(C)[rng.integers(0, C, size=N)], num_classes=C)
    # Upstream grad from CE is probs - y (assuming average over N done elsewhere)
    dY = probs - y
    dX = sm.backward(dY)
    print("Softmax grad shape:", dX.shape, "mean:", float(dX.mean()))


if __name__ == "__main__":
    demo_dense_and_activations()