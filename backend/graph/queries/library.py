"""Graph Cypher query definitions — single source of truth for backend graph."""

from __future__ import annotations

from typing import Any

from graph.entities.base import GraphEntity
from graph.entities.schemas import APPEND_ONLY_LABELS, NATURAL_KEYS
from graph.relationships import GraphRelationship


def user_timeline(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL "
        "RETURN s ORDER BY s.timestamp ASC",
        {"user_id": user_id},
    )


def session_history(user_id: str, limit: int = 12) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "WHERE s.deleted_at IS NULL "
        "RETURN s ORDER BY s.timestamp DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )


def risk_progression(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:PRODUCED|PREDICTED]->(p:Prediction) "
        "RETURN s.session_id AS session, "
        "coalesce(p.risk_probability, p.risk_score) AS risk, "
        "p.created_at AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def emotion_evolution(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:EXPRESSES|IDENTIFIED]->(e:Emotion) "
        "RETURN coalesce(e.emotion_name, e.emotion) AS emotion, "
        "e.probability AS prob, e.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def symptom_evolution(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "-[:INDICATES|IDENTIFIED]->(sym:Symptom) "
        "RETURN coalesce(sym.symptom_name, sym.symptom) AS symptom, "
        "sym.confidence AS conf, sym.timestamp AS ts ORDER BY ts ASC",
        {"user_id": user_id},
    )


def high_risk_users(threshold: float = 0.7) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (s:Session)-[:PREDICTED]->(p:Prediction) WHERE p.risk_score >= $threshold "
        "RETURN DISTINCT s.session_id AS session, p.risk_score AS risk ORDER BY risk DESC",
        {"threshold": threshold},
    )


def graph_statistics() -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS edges",
        {},
    )


def temporal_chain(user_id: str) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session) "
        "MATCH path=(s)-[:TEMPORALLY_PRECEDES*]->(s2:Session) RETURN path",
        {"user_id": user_id},
    )


def subgraph_extraction(node_id: str, depth: int = 2) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH path=(n {uuid: $node_id})-[*1..$depth]-(m) RETURN path LIMIT 500",
        {"node_id": node_id, "depth": depth},
    )


def merge_entity(entity: GraphEntity) -> tuple[str, dict[str, Any]]:
    """MERGE stable node by natural key — updates last_updated only."""
    props = entity.to_neo4j_props()
    key = NATURAL_KEYS.get(entity.label, "uuid")
    key_val = props.get(key, entity.identity.uuid)
    cypher = (
        f"MERGE (n:{entity.label} {{{key}: $key_val}}) "
        f"ON CREATE SET n += $props "
        f"ON MATCH SET n += $props, n.last_updated = $props.last_updated "
        f"RETURN n"
    )
    return cypher, {"key_val": key_val, "props": props}


def create_entity(entity: GraphEntity) -> tuple[str, dict[str, Any]]:
    """CREATE append-only node — never overwrites history."""
    props = entity.to_neo4j_props()
    cypher = f"CREATE (n:{entity.label}) SET n = $props RETURN n"
    return cypher, {"props": props}


def persist_entity(entity: GraphEntity) -> tuple[str, dict[str, Any]]:
    """Route to CREATE (append-only) or MERGE (stable) based on label."""
    if entity.label in APPEND_ONLY_LABELS:
        return create_entity(entity)
    return merge_entity(entity)


def soft_delete_entity(label: str, natural_key: str, key_value: str, deleted_at: str) -> tuple[str, dict[str, Any]]:
    return (
        f"MATCH (n:{label} {{{natural_key}: $key_val}}) "
        f"SET n.deleted_at = $deleted_at, n.last_updated = $deleted_at RETURN n",
        {"key_val": key_value, "deleted_at": deleted_at},
    )


def hard_delete_user_subgraph(user_id: str) -> tuple[str, dict[str, Any]]:
    """GDPR-style erasure — detach delete all user-owned nodes."""
    return (
        "MATCH (u:User {user_id: $user_id}) "
        "OPTIONAL MATCH (u)-[*]->(n) "
        "DETACH DELETE u, n",
        {"user_id": user_id},
    )


def merge_relationship(
    rel: GraphRelationship,
    source_label: str,
    target_label: str,
) -> tuple[str, dict[str, Any]]:
    return rel.to_cypher_merge(source_label, target_label)


def lookup_by_uuid(label: str, uuid: str) -> tuple[str, dict[str, Any]]:
    return (
        f"MATCH (n:{label} {{uuid: $uuid}}) RETURN n LIMIT 1",
        {"uuid": uuid},
    )


def recommendations_for_user(user_id: str, limit: int = 10) -> tuple[str, dict[str, Any]]:
    return (
        "MATCH (u:User {user_id: $user_id})-[:HAS_SESSION]->(s:Session)-[:RECOMMENDS]->(r:Recommendation) "
        "RETURN r ORDER BY r.timestamp DESC LIMIT $limit",
        {"user_id": user_id, "limit": limit},
    )
