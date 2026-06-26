"""Temporal graph builder — chains sessions with TEMPORALLY_PRECEDES."""

from __future__ import annotations

from typing import Any

from ml_pipeline.graph.builders.neo4j_builder import Neo4jGraphBuilder
from ml_pipeline.graph.schema.types import GraphEdge, GraphSnapshot


class TemporalGraphBuilder(Neo4jGraphBuilder):
    def __init__(self, store=None, max_sessions: int = 12) -> None:
        super().__init__(store)
        self.max_sessions = max_sessions
        self._session_chains: dict[str, list[str]] = {}

    def build(self, data: dict[str, Any]) -> GraphSnapshot:
        snap = super().build(data)
        user_id = str(data["participant_id"])
        session_id = str(data.get("session_id", user_id))
        chain = self._session_chains.setdefault(user_id, [])
        if chain:
            prev = chain[-1]
            self.store.add_edge(
                GraphEdge(
                    f"Session:{prev}",
                    f"Session:{session_id}",
                    "TEMPORALLY_PRECEDES",
                    {"weight": 1.0, "confidence": 1.0},
                )
            )
            self.store.add_edge(GraphEdge(f"Session:{prev}", f"Session:{session_id}", "FOLLOWED_BY"))
        chain.append(session_id)
        self._session_chains[user_id] = chain[-self.max_sessions :]
        return self.store.snapshot(metadata={"builder": "temporal", "chain_length": len(chain)})

    def link_history(self, user_id: str, session_ids: list[str]) -> None:
        """Link an ordered list of session IDs."""
        for i in range(1, len(session_ids)):
            src, tgt = session_ids[i - 1], session_ids[i]
            self.store.add_edge(
                GraphEdge(f"Session:{src}", f"Session:{tgt}", "TEMPORALLY_PRECEDES", {"weight": 1.0})
            )
        self._session_chains[user_id] = session_ids[-self.max_sessions :]
