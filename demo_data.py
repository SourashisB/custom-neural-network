import numpy as np

from src.data import load_iris_pandas, load_mnist_npz, load_mnist_csv, split_train_val_test


def demo_iris(csv_path: str):
    ds = load_iris_pandas(csv_path, val_ratio=0.2, test_ratio=0.1, seed=123)
    print("Iris shapes:")
    print("Train:", ds.X_train.shape, ds.y_train.shape)
    print("Val  :", ds.X_val.shape, ds.y_val.shape)
    if ds.X_test is not None:
        print("Test :", ds.X_test.shape, ds.y_test.shape)


def demo_mnist_npz(npz_path: str):
    X_tr, y_tr = load_mnist_npz(npz_path, split="train")
    X_te, y_te = load_mnist_npz(npz_path, split="test")
    print("MNIST npz:")
    print("Train:", X_tr.shape, y_tr.shape, "min/max", float(X_tr.min()), float(X_tr.max()))
    print("Test :", X_te.shape, y_te.shape)


def demo_mnist_csv(csv_path: str):
    X, y = load_mnist_csv(csv_path)
    ds = split_train_val_test(X, y, val_ratio=0.1, test_ratio=0.1, seed=7)
    print("MNIST csv splits:")
    print("Train:", ds.X_train.shape, "Val:", ds.X_val.shape, "Test:", ds.X_test.shape)


if __name__ == "__main__":
    # Provide local paths to run these demos:
    # demo_iris("path/to/iris.csv")
    # demo_mnist_npz("path/to/mnist.npz")
    # demo_mnist_csv("path/to/mnist.csv")
    pass