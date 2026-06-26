"""Neo4j graph store — optional driver with fail-open fallback."""

from __future__ import annotations

import logging
from typing import Any

from ml_pipeline.graph.config import graph_config
from ml_pipeline.graph.graph_store.memory_store import InMemoryGraphStore
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Persist graph to Neo4j when driver is available."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None, database: str | None = None) -> None:
        cfg = graph_config()
        neo = cfg.get("neo4j", {})
        self.uri = uri or neo.get("uri", "bolt://localhost:7687")
        self.user = user or neo.get("user", "neo4j")
        self.password = password or neo.get("password", "mindgraph")
        self.database = database or neo.get("database", "neo4j")
        self._driver = None
        self._fallback = InMemoryGraphStore()
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
        except Exception as exc:
            logger.warning("Neo4j unavailable, using in-memory fallback: %s", exc)
            self._driver = None

    @property
    def available(self) -> bool:
        return self._driver is not None

    def upsert_node(self, node: GraphNode) -> GraphNode:
        if not self._driver:
            return self._fallback.upsert_node(node)
        props = {**node.properties, "node_id": node.node_id, "version": node.version}
        cypher = f"MERGE (n:{node.label} {{node_id: $node_id}}) SET n += $props RETURN n"
        with self._driver.session(database=self.database) as session:
            session.run(cypher, node_id=node.node_id, props=props)
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        e = edge.with_defaults()
        if not self._driver:
            return self._fallback.add_edge(e)
        cypher = (
            f"MATCH (a {{node_id: $src}}), (b {{node_id: $tgt}}) "
            f"MERGE (a)-[r:{e.rel_type}]->(b) SET r += $props RETURN r"
        )
        with self._driver.session(database=self.database) as session:
            session.run(cypher, src=e.source_id, tgt=e.target_id, props=e.properties)
        return e

    def run_cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not self._driver:
            return []
        with self._driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [dict(r) for r in result]

    def snapshot(self, version: str = "1.0.0", metadata: dict[str, Any] | None = None) -> GraphSnapshot:
        if not self._driver:
            return self._fallback.snapshot(version, metadata)
        nodes_raw = self.run_cypher("MATCH (n) RETURN n.node_id AS node_id, labels(n) AS labels, properties(n) AS props")
        edges_raw = self.run_cypher(
            "MATCH (a)-[r]->(b) RETURN a.node_id AS src, b.node_id AS tgt, type(r) AS rel, properties(r) AS props"
        )
        nodes = [
            GraphNode(node_id=r["node_id"], label=r["labels"][0] if r["labels"] else "Unknown", properties=r["props"] or {})
            for r in nodes_raw
            if r.get("node_id")
        ]
        edges = [
            GraphEdge(r["src"], r["tgt"], r["rel"], r["props"] or {})
            for r in edges_raw
            if r.get("src") and r.get("tgt")
        ]
        return GraphSnapshot(nodes=nodes, edges=edges, version=version, metadata=metadata or {})

    def close(self) -> None:
        if self._driver:
            self._driver.close()
