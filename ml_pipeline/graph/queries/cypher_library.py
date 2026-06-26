"""Reusable Cypher query library."""

from __future__ import annotations


def user_timeline(user_id: str) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "RETURN s ORDER BY s.timestamp ASC",
        {"user_id": user_id},
    )


def session_history(user_id: str, limit: int = 12) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "RETURN s ORDER BY s.timestamp DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def risk_progression(user_id: str) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:PREDICTED]->(p:Prediction) "
        "RETURN s.session_id AS session, p.risk_score AS risk, p.created_at AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def emotion_evolution(user_id: str) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:EXPRESSES]->(e:Emotion) "
        "RETURN e.emotion AS emotion, e.probability AS prob, e.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def symptom_evolution(user_id: str) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:INDICATES]->(sym:Symptom) "
        "RETURN sym.symptom AS symptom, sym.confidence AS conf, sym.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def high_risk_users(threshold: float = 0.7) -> tuple[str, dict]:
    return (
        "MATCH (s:Session)-[:PREDICTED]->(p:Prediction) WHERE p.risk_score >= $threshold "
        "RETURN DISTINCT s.session_id AS session, p.risk_score AS risk ORDER BY risk DESC",
        {"threshold": threshold},
    )


def graph_statistics() -> tuple[str, dict]:
    return (
        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS edges",
        {},
    )


def temporal_chain(user_id: str) -> tuple[str, dict]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "MATCH path=(s)-[:TEMPORALLY_PRECEDES*]->(s2:Session) RETURN path",
        {"user_id": user_id},
    )


def subgraph_extraction(node_id: str, depth: int = 2) -> tuple[str, dict]:
    return (
        "MATCH path=(n {node_id: $node_id})-[*1..$depth]-(m) RETURN path",
        {"node_id": node_id, "depth": depth},
    )
