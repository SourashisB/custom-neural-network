# examples/demo_utils.py

import numpy as np

from src.logger import CSVLogger, StdoutLogger
from src.utils import (
    batch_iterator,
    one_hot,
    set_seed,
    shuffle_in_unison,
    train_val_split,
)

def main():
    set_seed(42)

    # Dummy data
    N, D, C = 23, 5, 3
    X = np.random.randn(N, D).astype(np.float32)
    y_idx = np.random.randint(0, C, size=N)
    Y = one_hot(y_idx, C)

    # Shuffle
    Xs, Ys = shuffle_in_unison(X, Y, seed=7)

    # Train/val split
    X_tr, Y_tr, X_val, Y_val = train_val_split(Xs, Ys, val_ratio=0.3, seed=123)

    # Iterate batches
    slog = StdoutLogger()
    slog.log(f"Train size: {X_tr.shape[0]}  | Val size: {X_val.shape[0]}")
    for xb, yb in batch_iterator(X_tr, Y_tr, batch_size=8, shuffle=True, seed=999):
        slog.log(f"Batch: X {xb.shape}, Y {yb.shape}")

    # CSV logging
    csv = CSVLogger("experiments/runs/demo/log.csv")
    for epoch in range(3):
        csv.log({"epoch": epoch + 1, "train_loss": float(np.random.rand()), "val_acc": float(np.random.rand())})
    csv.close()
    slog.log("Wrote logs to experiments/runs/demo/log.csv")

if __name__ == "__main__":
    main()