"""Backward-compatible logging imports. Prefer ``app.logging``."""

from app.logging import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
