"""Graph platform monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMetrics:
    operation: str
    execution_time_ms: float
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphMonitor:
    """Track query latency against performance targets."""

    TARGETS_MS: dict[str, float] = {
        "insert": 50,
        "lookup": 10,
        "relationship": 20,
        "timeline": 50,
        "recommendation": 100,
    }

    def __init__(self) -> None:
        self._metrics: list[QueryMetrics] = []

    def record(self, operation: str, execution_time_ms: float, *, success: bool = True, **metadata: Any) -> QueryMetrics:
        metric = QueryMetrics(operation=operation, execution_time_ms=execution_time_ms, success=success, metadata=metadata)
        self._metrics.append(metric)
        return metric

    def check_sla(self, operation: str, execution_time_ms: float) -> bool:
        target = self.TARGETS_MS.get(operation)
        if target is None:
            return True
        return execution_time_ms <= target

    def summary(self) -> dict[str, Any]:
        if not self._metrics:
            return {"count": 0}
        by_op: dict[str, list[float]] = {}
        for m in self._metrics:
            by_op.setdefault(m.operation, []).append(m.execution_time_ms)
        return {
            "count": len(self._metrics),
            "by_operation": {op: {"avg_ms": sum(vals) / len(vals), "max_ms": max(vals)} for op, vals in by_op.items()},
        }

    class Timer:
        def __init__(self, monitor: GraphMonitor, operation: str) -> None:
            self._monitor = monitor
            self._operation = operation
            self._start = 0.0

        def __enter__(self) -> GraphMonitor.Timer:
            self._start = time.perf_counter()
            return self

        def __exit__(self, *args: object) -> None:
            elapsed = (time.perf_counter() - self._start) * 1000
            self._monitor.record(self._operation, elapsed)
