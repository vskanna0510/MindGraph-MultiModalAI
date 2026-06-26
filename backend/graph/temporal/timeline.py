"""Timeline unit helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum


class TimelineUnit(str, Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


_UNIT_DELTAS: dict[TimelineUnit, timedelta] = {
    TimelineUnit.SECONDS: timedelta(seconds=1),
    TimelineUnit.MINUTES: timedelta(minutes=1),
    TimelineUnit.HOURS: timedelta(hours=1),
    TimelineUnit.DAYS: timedelta(days=1),
    TimelineUnit.WEEKS: timedelta(weeks=1),
    TimelineUnit.MONTHS: timedelta(days=30),
    TimelineUnit.YEARS: timedelta(days=365),
}


def parse_timeline_window(window_days: int = 30) -> tuple[str, str]:
    """Return ISO start/end for absolute timeline window."""
    end = datetime.now(UTC)
    start = end - timedelta(days=window_days)
    return start.isoformat(), end.isoformat()
