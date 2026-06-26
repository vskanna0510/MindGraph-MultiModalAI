"""Database connectivity health probes."""

from __future__ import annotations

import asyncio

from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.settings import Settings


async def check_postgres(settings: Settings) -> str:
    """Return postgres health status."""
    try:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_redis(settings: Settings) -> str:
    """Return redis health status."""
    try:
        client = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        pong = await client.ping()
        await client.aclose()
        return "healthy" if pong else "unhealthy"
    except Exception:
        return "unhealthy"


async def check_neo4j(settings: Settings) -> str:
    """Return neo4j health status."""
    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run("RETURN 1")
        await driver.close()
        return "healthy"
    except Exception:
        return "unhealthy"


async def check_all(settings: Settings, timeout: float = 5.0) -> dict[str, str]:
    """Check all dependencies with a global timeout."""

    async def _run() -> dict[str, str]:
        postgres, redis_status, neo4j = await asyncio.gather(
            check_postgres(settings),
            check_redis(settings),
            check_neo4j(settings),
        )
        return {"postgres": postgres, "redis": redis_status, "neo4j": neo4j}

    try:
        return await asyncio.wait_for(_run(), timeout=timeout)
    except TimeoutError:
        return {"postgres": "timeout", "redis": "timeout", "neo4j": "timeout"}
