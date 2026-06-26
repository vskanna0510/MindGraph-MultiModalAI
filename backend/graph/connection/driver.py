"""Async Neo4j driver singleton."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from graph.config.loader import get_graph_config

if TYPE_CHECKING:
    from app.config.settings import Settings

_driver: AsyncDriver | None = None


def init_neo4j_driver(
    uri: str,
    username: str,
    password: str,
    *,
    max_pool_size: int | None = None,
) -> AsyncDriver:
    """Create or return the shared async Neo4j driver."""
    global _driver
    if _driver is None:
        cfg = get_graph_config()
        pool = max_pool_size or cfg.pool_size
        _driver = AsyncGraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_pool_size=pool,
        )
    return _driver


def get_neo4j_driver() -> AsyncDriver:
    if _driver is None:
        from app.config.settings import get_settings

        settings = get_settings()
        return init_neo4j_driver(
            settings.neo4j_uri,
            settings.neo4j_username,
            settings.neo4j_password,
        )
    return _driver


async def close_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def get_neo4j_session(
    database: str | None = None,
    settings: Settings | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a Neo4j async session."""
    if settings is None:
        from app.config.settings import get_settings

        settings = get_settings()
    driver = get_neo4j_driver()
    db = database or settings.neo4j_database
    async with driver.session(database=db) as session:
        yield session
