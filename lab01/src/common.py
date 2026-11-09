"""Shared helpers for Lab 1: seeding, data split, model factory, memory sampling."""
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is part of the stack but not needed for this lab
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split, random_state=42."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_models():
    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


class PeakRSS:
    """Context manager that samples process RSS in a background thread.

    Reports the baseline RSS before the block, the peak RSS during it, and the
    increase (peak - baseline). Uses psutil.Process().memory_info().rss.
    """

    def __init__(self, interval: float = 0.001):
        self.interval = interval
        self.proc = psutil.Process()
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def delta(self):
        return self.peak - self.baseline
