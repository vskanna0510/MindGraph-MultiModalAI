"""Graph integrity validation."""

from __future__ import annotations

from collections import Counter

from ml_pipeline.graph.schema.types import GraphSnapshot


class GraphValidator:
    def validate(self, snapshot: GraphSnapshot) -> tuple[bool, list[str]]:
        errors: list[str] = []
        node_ids = [n.node_id for n in snapshot.nodes]
        dupes = [k for k, v in Counter(node_ids).items() if v > 1]
        if dupes:
            errors.append(f"duplicate_nodes:{dupes[:3]}")
        known = set(node_ids)
        for e in snapshot.edges:
            if e.source_id not in known:
                errors.append(f"missing_source:{e.source_id}")
            if e.target_id not in known:
                errors.append(f"missing_target:{e.target_id}")
        sessions = [n.node_id for n in snapshot.nodes if n.label == "Session"]
        temporal_edges = [e for e in snapshot.edges if e.rel_type == "TEMPORALLY_PRECEDES"]
        if len(sessions) > 1 and not temporal_edges:
            errors.append("disconnected_session_chain")
        if not snapshot.nodes:
            errors.append("empty_graph")
        return len(errors) == 0, errors
