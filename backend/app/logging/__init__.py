"""Centralized structured logging for MindGraph++.

Purpose:
    Configure structlog and provide module-level loggers with correlation IDs.

Dependencies:
    - structlog
    - app.config.settings.Settings

Usage:
    from app.logging import configure_logging, get_logger
    logger = get_logger(__name__)
    logger.info("event_name", key="value")
"""

import logging
import sys
from typing import Any

import structlog

from app.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog and standard library logging.

    Parameters:
        settings: Application settings containing log level and format.

    Side Effects:
        Reconfigures global logging and structlog processors.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    use_json = settings.log_format.lower() == "json"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to a module name.

    Parameters:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)
