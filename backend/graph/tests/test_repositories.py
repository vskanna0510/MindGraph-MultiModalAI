"""Graph repository tests with mocked Neo4j session."""

from __future__ import annotations

import pytest

from graph.entities.factories import UserFactory
from graph.repositories import UserRepository
from graph.tests.conftest import MockAsyncSession


@pytest.mark.asyncio
async def test_user_repository_save() -> None:
    session = MockAsyncSession()
    session.queue([{"n": {"user_id": "u1", "uuid": "abc"}}])
    repo = UserRepository(session)  # type: ignore[arg-type]
    entity = UserFactory.create("u1", owner="u1")
    saved = await repo.save(entity)
    assert saved.get("user_id") == "u1" or "user_id" in str(saved)


@pytest.mark.asyncio
async def test_user_repository_get_missing() -> None:
    session = MockAsyncSession()
    session.queue([])
    repo = UserRepository(session)  # type: ignore[arg-type]
    result = await repo.get_by_id("missing")
    assert result is None
