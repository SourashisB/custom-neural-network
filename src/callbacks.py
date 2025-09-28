from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class Callback:
    def on_train_begin(self): ...
    def on_epoch_begin(self, epoch: int): ...
    def on_batch_end(self, batch: int, logs: Dict): ...
    def on_epoch_end(self, epoch: int, logs: Dict): ...
    def on_train_end(self): ...


@dataclass
class EarlyStopping(Callback):
    monitor: str = "val_loss"
    patience: int = 10
    mode: str = "min"  # "min" or "max"
    best: Optional[float] = None
    wait: int = 0
    stopped_epoch: Optional[int] = None
    stop_training: bool = False

    def on_train_begin(self):
        self.best = None
        self.wait = 0
        self.stopped_epoch = None
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict):
        value = logs.get(self.monitor)
        if value is None:
            return
        better = (value < self.best) if (self.best is not None and self.mode == "min") else \
                 (value > self.best) if (self.best is not None and self.mode == "max") else True
        if better:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


@dataclass
class ReduceLROnPlateau(Callback):
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: int = 5
    min_lr: float = 0.0
    mode: str = "min"
    best: Optional[float] = None
    wait: int = 0

    def __init__(self, optimizer, monitor="val_loss", factor=0.1, patience=5, min_lr=0.0, mode="min"):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.best = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Dict):
        value = logs.get(self.monitor)
        if value is None:
            return
        better = (value < self.best) if (self.best is not None and self.mode == "min") else \
                 (value > self.best) if (self.best is not None and self.mode == "max") else True
        if better:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce lr
                if hasattr(self.optimizer, "lr"):
                    new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                    self.optimizer.lr = new_lr
                self.wait = 0