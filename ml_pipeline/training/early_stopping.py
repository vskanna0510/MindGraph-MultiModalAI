"""Early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 10
    monitor: str = "val_f1"
    mode: str = "max"
    counter: int = 0
    best_score: float | None = None
    should_stop: bool = False

    def step(self, metrics: dict[str, float]) -> bool:
        if self.monitor not in metrics:
            return False
        score = metrics[self.monitor]
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
