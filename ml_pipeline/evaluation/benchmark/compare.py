"""Model comparison and benchmarking."""

from __future__ import annotations

import time
from typing import Any

import torch


def benchmark_inference(model, batch, runs: int = 20, warmup: int = 3) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        times = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(runs):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "inference_time_ms": sum(times) / len(times),
        "params": param_count,
        "model_size_mb": param_count * 4 / (1024 * 1024),
    }


def compare_models(results: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in results.items():
        rows.append({"model": name, **metrics})
    rows.sort(key=lambda r: r.get("f1", r.get("roc_auc", 0)), reverse=True)
    return rows
