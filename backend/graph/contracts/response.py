"""Standard graph service response contract."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class GraphServiceResponse(Generic[T]):
    """Every graph service returns result + metadata per MP4 Part 3."""

    result: T
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    version: str = "2.0.0"
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "version": self.version,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "correlation_id": self.correlation_id,
            "cache_hit": self.cache_hit,
        }


class TimedOperation:
    """Context manager for timing graph operations."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> TimedOperation:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
