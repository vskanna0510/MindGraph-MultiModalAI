"""Parallel processing utilities for dataset pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

from ml_pipeline.datasets.config import dataset_config
from ml_pipeline.datasets.logging_utils import processing_logger

T = TypeVar("T")
R = TypeVar("R")


def worker_count() -> int:
    cfg = dataset_config().get("multiprocessing", {})
    configured = int(cfg.get("max_workers", 0))
    if configured <= 0:
        return max(1, (os.cpu_count() or 1) - 1)
    return configured


def map_parallel(
    func: Callable[[T], R],
    items: list[T],
    description: str = "task",
) -> tuple[list[R], list[tuple[T, Exception]]]:
    """Run func over items with graceful per-item failure recovery."""
    cfg = dataset_config().get("multiprocessing", {})
    if not cfg.get("enabled", True) or len(items) <= 1:
        results: list[R] = []
        failures: list[tuple[T, Exception]] = []
        for item in items:
            try:
                results.append(func(item))
            except Exception as exc:
                failures.append((item, exc))
                processing_logger.error("%s_failed item=%s error=%s", description, item, exc)
        return results, failures

    results = []
    failures: list[tuple[T, Exception]] = []
    workers = min(worker_count(), len(items))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(func, item): item for item in items}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((item, exc))
                processing_logger.error("%s_failed item=%s error=%s", description, item, exc)
    return results, failures
