"""Regression metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[valid], target[valid]
    if len(pred) == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    medae = float(np.median(np.abs(err)))
    mape = float(np.mean(np.abs(err / (target + 1e-8)))) if len(target) else 0.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot
    explained_var = float(1 - np.var(target - pred) / (np.var(target) + 1e-12))
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2, "explained_variance": explained_var, "median_ae": medae}
