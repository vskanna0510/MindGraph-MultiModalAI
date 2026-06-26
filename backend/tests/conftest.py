"""Pytest configuration and shared fixtures."""

import os

import pytest

os.environ.setdefault("SKIP_STARTUP_VALIDATION", "true")


@pytest.fixture(scope="session", autouse=True)
def _skip_startup_validation() -> None:
    os.environ["SKIP_STARTUP_VALIDATION"] = "true"
