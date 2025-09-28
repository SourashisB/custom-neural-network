# src/logger.py

import csv
import os
import sys
import time
from typing import Dict, Optional, TextIO


class CSVLogger:
    """
    Minimal CSV logger that writes metrics per step/epoch.

    Usage:
        logger = CSVLogger(filepath="experiments/runs/run1/log.csv")
        logger.log({"epoch": 1, "train_loss": 0.52, "val_acc": 0.81})
        logger.close()
    """

    def __init__(self, filepath: str, append: bool = False, flush: bool = True):
        self.filepath = filepath
        self.append = append
        self.flush = flush
        self._fieldnames = None  # type: Optional[list[str]]
        self._fh = None  # type: Optional[TextIO]
        self._writer = None  # csv.DictWriter
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mode = "a" if append and os.path.exists(filepath) else "w"
        self._fh = open(filepath, mode, newline="", encoding="utf-8")
        self._writer = None

    def log(self, row: Dict):
        if self._writer is None:
            # Initialize writer with the first row's keys (ordered)
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            # Ensure consistent columns; fill missing keys with None
            for k in self._fieldnames:
                if k not in row:
                    row[k] = None
            # Add new keys if needed (rare; safer to keep stable schema)
            extra_keys = [k for k in row.keys() if k not in self._fieldnames]
            if extra_keys:
                # Rebuild file with new header (simple approach: close and raise)
                raise ValueError(
                    f"CSVLogger received unknown keys after initialization: {extra_keys}. "
                    f"Keep a stable set of columns per run."
                )

        self._writer.writerow(row)
        if self.flush:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class StdoutLogger:
    """
    Simple stdout logger with timestamp prefix.

    Usage:
        slog = StdoutLogger()
        slog.log("Epoch 1 | loss=0.52 | acc=0.81")
    """

    def __init__(self, stream: TextIO = sys.stdout, timefmt: str = "%Y-%m-%d %H:%M:%S"):
        self.stream = stream
        self.timefmt = timefmt

    def log(self, msg: str):
        ts = time.strftime(self.timefmt, time.localtime())
        self.stream.write(f"[{ts}] {msg}\n")
        self.stream.flush()