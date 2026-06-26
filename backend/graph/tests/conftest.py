"""Fixtures for graph platform tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockResult:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record


class MockAsyncSession:
    """Minimal async Neo4j session mock."""

    def __init__(self) -> None:
        self.run = AsyncMock(side_effect=self._default_run)
        self.execute_write = AsyncMock(side_effect=self._execute_write)
        self._responses: list[dict[str, Any]] = [{"n": {"user_id": "u1"}}]

    async def _default_run(self, cypher: str, params: dict | None = None) -> MockResult:
        return MockResult(self._responses)

    async def _execute_write(self, fn: Any) -> list[dict[str, Any]]:
        tx = MagicMock()
        tx.run = self.run
        return await fn(tx)

    def queue(self, records: list[dict[str, Any]]) -> None:
        self._responses = records


@pytest.fixture
def mock_neo4j_session() -> MockAsyncSession:
    return MockAsyncSession()
