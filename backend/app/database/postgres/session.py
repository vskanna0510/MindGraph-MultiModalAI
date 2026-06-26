"""SQLAlchemy declarative base and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import Settings, get_settings


class Base(DeclarativeBase):
    """SQLAlchemy ORM base class."""


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None):
    """Return or create the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        resolved = settings or get_settings()
        _engine = create_async_engine(
            resolved.database_url,
            pool_pre_ping=True,
            echo=resolved.is_development,
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(settings),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session
