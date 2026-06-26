"""Repository layer contracts.

Repositories encapsulate database and cache access. They must not contain
business rules — only persistence operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
ID = TypeVar("ID")


class Repository(ABC, Generic[T, ID]):
    """Abstract base repository for CRUD-style persistence."""

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> T | None:
        """Retrieve an entity by identifier."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Persist an entity and return the stored representation."""

    @abstractmethod
    async def delete(self, entity_id: ID) -> None:
        """Remove an entity by identifier."""
