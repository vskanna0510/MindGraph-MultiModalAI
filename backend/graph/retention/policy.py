"""Data retention and GDPR-style erasure policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from neo4j import AsyncSession

from graph.queries.library import hard_delete_user_subgraph, soft_delete_entity
from graph.repositories.base import BaseGraphRepository
from graph.schema.nodes.definitions import NATURAL_KEYS


class RetentionAction(str, Enum):
    SOFT_DELETE = "soft_delete"
    HARD_DELETE = "hard_delete"
    ANONYMIZE = "anonymize"
    CONSENT_WITHDRAWAL = "consent_withdrawal"


@dataclass
class RetentionResult:
    action: str
    user_id: str
    affected: int = 0
    timestamp: str = ""


class DataRetentionPolicy:
    """Support soft delete, hard delete, GDPR erasure, consent withdrawal."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = BaseGraphRepository(session)

    async def soft_delete_session(self, user_id: str, session_id: str) -> RetentionResult:
        now = datetime.now(UTC).isoformat()
        cypher, params = soft_delete_entity("Session", "session_id", session_id, now)
        records, _ = await self._repo._write(cypher, params)
        return RetentionResult(
            action=RetentionAction.SOFT_DELETE.value,
            user_id=user_id,
            affected=len(records),
            timestamp=now,
        )

    async def hard_delete_user(self, user_id: str) -> RetentionResult:
        cypher, params = hard_delete_user_subgraph(user_id)
        await self._repo._write(cypher, params)
        return RetentionResult(
            action=RetentionAction.HARD_DELETE.value,
            user_id=user_id,
            affected=1,
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def anonymize_user(self, user_id: str) -> RetentionResult:
        """Replace PII with anonymized placeholders — preserve research structure."""
        cypher = (
            "MATCH (u:User {user_id: $user_id}) "
            "SET u.external_id = 'anon_' + u.uuid, "
            "u.preferred_language = 'redacted', "
            "u.country = 'redacted', "
            "u.consent_status = 'anonymized', "
            "u.anonymized_at = $ts "
            "RETURN u"
        )
        ts = datetime.now(UTC).isoformat()
        records, _ = await self._repo._write(cypher, {"user_id": user_id, "ts": ts})
        return RetentionResult(
            action=RetentionAction.ANONYMIZE.value,
            user_id=user_id,
            affected=len(records),
            timestamp=ts,
        )

    async def consent_withdrawal(self, user_id: str) -> RetentionResult:
        cypher = (
            "MATCH (u:User {user_id: $user_id}) "
            "SET u.consent_status = 'withdrawn', u.consent_withdrawn_at = $ts "
            "RETURN u"
        )
        ts = datetime.now(UTC).isoformat()
        records, _ = await self._repo._write(cypher, {"user_id": user_id, "ts": ts})
        return RetentionResult(
            action=RetentionAction.CONSENT_WITHDRAWAL.value,
            user_id=user_id,
            affected=len(records),
            timestamp=ts,
        )
