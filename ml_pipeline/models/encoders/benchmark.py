"""Independent encoder benchmarking."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


def benchmark_encoder(
    encoder: nn.Module,
    sample_input: torch.Tensor,
    mask: torch.Tensor | None = None,
    device: torch.device | None = None,
    runs: int = 10,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    encoder = encoder.to(device)
    sample_input = sample_input.to(device)
    if mask is not None:
        mask = mask.to(device)
    encoder.eval()
    param_count = sum(p.numel() for p in encoder.parameters())
    times: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            if hasattr(encoder, "forward") and mask is not None:
                encoder(sample_input, mask)
            else:
                encoder(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    mem_mb = 0.0
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "mean_latency_ms": sum(times) / len(times) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
        "parameters": param_count,
        "model_size_mb": param_count * 4 / (1024 ** 2),
        "gpu_memory_mb": mem_mb,
        "device": str(device),
    }
