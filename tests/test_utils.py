# tests/test_utils.py

import numpy as np

from src.utils import batch_iterator, one_hot, set_seed, shuffle_in_unison, train_val_split


def test_one_hot_basic():
    y = np.array([0, 2, 1, 2])
    Y = one_hot(y, 3)
    assert Y.shape == (4, 3)
    assert (Y.argmax(axis=1) == y).all()


def test_shuffle_in_unison_deterministic():
    set_seed(0)
    X = np.arange(10)[:, None]
    y = np.arange(10)
    Xs1, ys1 = shuffle_in_unison(X, y, seed=123)
    Xs2, ys2 = shuffle_in_unison(X, y, seed=123)
    assert np.array_equal(Xs1, Xs2)
    assert np.array_equal(ys1, ys2)


def test_batch_iterator_shapes():
    N, D = 17, 4
    X = np.random.randn(N, D)
    y = np.random.randint(0, 3, size=N)
    bs = 5
    total = 0
    for xb, yb in batch_iterator(X, y, batch_size=bs, shuffle=False, drop_last=False):
        assert xb.shape[0] <= bs
        assert yb.shape[0] == xb.shape[0]
        total += xb.shape[0]
    assert total == N


def test_train_val_split_ratio():
    X = np.arange(20)[:, None]
    y = np.arange(20)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.25, seed=42)
    assert X_tr.shape[0] == 15
    assert X_val.shape[0] == 5
    assert y_tr.shape[0] == 15
    assert y_val.shape[0] == 5