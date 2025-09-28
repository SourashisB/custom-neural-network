# src/training.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .logger import CSVLogger, StdoutLogger
from .metrics import accuracy, confusion_matrix
from .utils import batch_iterator, train_val_split


@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 32
    shuffle: bool = True
    val_ratio: float = 0.2  # if X_val/y_val not provided
    task: str = "multiclass"  # "multiclass" or "binary"
    threshold: float = 0.5
    log_to_csv: Optional[str] = None  # filepath or None
    seed: Optional[int] = 42


class Trainer:
    """
    Minimal training loop runner.

    Expects:
      - model: has forward, backward, parameters, zero_grad, predict_proba/predict (optional)
      - optimizer: has step(params_iter)
      - loss: has forward_backward(y_pred, y_true, backward=True)
    """

    def __init__(self, model, optimizer, loss, config: TrainerConfig = TrainerConfig()):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.cfg = config
        self.slog = StdoutLogger()
        self.csv = CSVLogger(self.cfg.log_to_csv) if self.cfg.log_to_csv else None
        self.callbacks = []  # type: List[object]

    def add_callback(self, cb) -> None:
        self.callbacks.append(cb)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        # Setup callbacks
        for cb in self.callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin()

        # Validation split if not provided
        if X_val is None or y_val is None:
            X_tr, y_tr, X_val, y_val = train_val_split(
                X, y, val_ratio=self.cfg.val_ratio, seed=self.cfg.seed, shuffle=True
            )
        else:
            X_tr, y_tr = X, y

        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.cfg.epochs + 1):
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_begin"):
                    cb.on_epoch_begin(epoch)

            # Training epoch
            losses = []
            correct = 0
            total = 0

            for bi, (xb, yb) in enumerate(
                batch_iterator(X_tr, y_tr, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle, seed=self.cfg.seed)
            ):
                logits_or_preds = self.model.forward(xb, training=True)
                loss, dY = self.loss.forward_backward(logits_or_preds, yb, backward=True)
                losses.append(loss)
                # Backward + step
                self.model.backward(dY)
                self.optimizer.step(self.model.parameters())
                self.model.zero_grad()

                # Train accuracy
                acc_batch = self._batch_accuracy(logits_or_preds, yb)
                correct += acc_batch * xb.shape[0]
                total += xb.shape[0]

                # Callbacks batch end
                logs_batch = {"loss": loss}
                for cb in self.callbacks:
                    if hasattr(cb, "on_batch_end"):
                        cb.on_batch_end(bi, logs_batch)

            train_loss = float(np.mean(losses) if losses else 0.0)
            train_acc = float(correct / max(total, 1))

            # Validation
            val_logits_or_preds = self.model.forward(X_val, training=False)
            val_loss, _ = self.loss.forward_backward(val_logits_or_preds, y_val, backward=False)
            val_acc = self._batch_accuracy(val_logits_or_preds, y_val)

            # Log
            history["loss"].append(train_loss)
            history["acc"].append(train_acc)
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))

            msg = f"Epoch {epoch:03d} | loss={train_loss:.4f} | acc={train_acc:.3f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
            self.slog.log(msg)
            if self.csv:
                self.csv.log({"epoch": epoch, "loss": train_loss, "acc": train_acc, "val_loss": float(val_loss), "val_acc": float(val_acc)})

            # Callbacks epoch end
            logs_epoch = {"loss": train_loss, "acc": train_acc, "val_loss": float(val_loss), "val_acc": float(val_acc)}
            stop = False
            for cb in self.callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, logs_epoch)
                if getattr(cb, "stop_training", False):
                    stop = True
            if stop:
                self.slog.log("Early stopping triggered.")
                break

        for cb in self.callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end()
        if self.csv:
            self.csv.close()

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        logits_or_preds = self.model.forward(X, training=False)
        loss, _ = self.loss.forward_backward(logits_or_preds, y, backward=False)
        acc = self._batch_accuracy(logits_or_preds, y)
        return {"loss": float(loss), "acc": float(acc)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.forward(X, training=False)

    def predict(
        self,
        X: np.ndarray,
        task: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        task = task or self.cfg.task
        threshold = threshold if threshold is not None else self.cfg.threshold
        scores = self.predict_proba(X)
        if task == "multiclass":
            return scores.argmax(axis=1)
        # binary
        scores = scores.squeeze()
        return (scores >= threshold).astype(int)

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        probs = self.predict_proba(X)
        return confusion_matrix(probs, y, num_classes=num_classes)

    def _batch_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.cfg.task == "multiclass":
            return accuracy(y_pred, y_true, task="multiclass")
        return accuracy(y_pred, y_true, task="binary", threshold=self.cfg.threshold)