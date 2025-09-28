# tests/test_losses.py

import numpy as np

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


def test_mse_gradcheck():
    rng = np.random.default_rng(0)
    y_pred = rng.standard_normal((4, 3)).astype("float64")
    y_true = rng.standard_normal((4, 3)).astype("float64")
    mse = MSELoss()

    def f(x):
        loss, _ = mse.forward_backward(x, y_true, backward=False)
        return loss

    loss, grad = mse.forward_backward(y_pred, y_true)
    num_grad = numerical_gradient(f, y_pred.copy())
    rel = np.linalg.norm(grad - num_grad) / (np.linalg.norm(grad) + np.linalg.norm(num_grad) + 1e-12)
    assert rel < 1e-6


def test_bce_logits_gradshape_and_values():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((7, 1)).astype("float64")
    y = (rng.random((7, 1)) > 0.5).astype("float64")
    bce = BinaryCrossEntropy(from_logits=True)

    def f(x):
        loss, _ = bce.forward_backward(x, y, backward=False)
        return loss

    loss, grad = bce.forward_backward(z, y)
    assert grad.shape == z.shape

    num_grad = numerical_gradient(f, z.copy())
    rel = np.linalg.norm(grad - num_grad) / (np.linalg.norm(grad) + np.linalg.norm(num_grad) + 1e-10)
    assert rel < 1e-4  # BCE is a bit trickier; loosen tolerance


def test_cce_logits_prob_equivalence():
    # When using logits with softmax, loss should match prob-mode if we pass the softmax probs
    rng = np.random.default_rng(2)
    N, C = 8, 5
    z = rng.standard_normal((N, C)).astype("float64")
    y_idx = rng.integers(0, C, size=N)

    cce_logits = CategoricalCrossEntropy(from_logits=True)
    loss_z, _ = cce_logits.forward_backward(z, y_idx, backward=False)

    # Convert to probabilities
    z_shift = z - z.max(axis=1, keepdims=True)
    p = np.exp(z_shift)
    p /= p.sum(axis=1, keepdims=True)

    cce_probs = CategoricalCrossEntropy(from_logits=False)
    loss_p, _ = cce_probs.forward_backward(p, y_idx, backward=False)

    assert np.allclose(loss_z, loss_p, atol=1e-10)


def test_cce_grad_properties_rowsum_zero():
    # Grad wrt logits should sum to zero per row (softmax-CCE)
    rng = np.random.default_rng(3)
    N, C = 6, 4
    z = rng.standard_normal((N, C)).astype("float64")
    y_idx = rng.integers(0, C, size=N)

    cce = CategoricalCrossEntropy(from_logits=True)
    _, grad = cce.forward_backward(z, y_idx)
    rowsum = grad.sum(axis=1)
    assert np.allclose(rowsum, 0.0, atol=1e-12)