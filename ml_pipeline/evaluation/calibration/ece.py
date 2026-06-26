"""Calibration metrics and scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    bins: list[dict]
    temperature: float = 1.0


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    labels = labels.astype(int)
    confidences = probs.max(axis=1) if probs.ndim == 2 else probs
    predictions = probs.argmax(axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = float((predictions[mask] == labels[mask]).mean())
        conf = float(confidences[mask].mean())
        gap = abs(acc - conf)
        ece += gap * mask.mean()
        mce = max(mce, gap)
        bins.append({"bin": i, "accuracy": acc, "confidence": conf, "count": int(mask.sum())})
    return CalibrationResult(ece=float(ece), mce=float(mce), bins=bins)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        self.train()
        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(logits), labels.long())
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.detach())


def calibrate_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    with torch.no_grad():
        calibrated = scaler(logits)
    return calibrated, t
