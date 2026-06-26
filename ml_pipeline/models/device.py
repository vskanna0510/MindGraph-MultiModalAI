"""Device and mixed-precision utilities."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def resolve_dtype(mixed_precision: str = "auto", device: torch.device | None = None) -> torch.dtype:
    device = device or resolve_device()
    if mixed_precision == "fp16" and device.type == "cuda":
        return torch.float16
    if mixed_precision == "bf16" and device.type == "cuda":
        return torch.bfloat16
    if mixed_precision == "auto" and device.type == "cuda":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()
