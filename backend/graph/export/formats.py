"""Multi-format graph export."""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import escape


class GraphExportEngine:
    """Export twin and graph data to multiple formats."""

    @staticmethod
    def to_json(data: Any) -> str:
        return json.dumps(data, default=str, indent=2)

    @staticmethod
    def to_csv(records: list[dict[str, Any]]) -> str:
        if not records:
            return ""
        headers = list(records[0].keys())
        lines = [",".join(headers)]
        for row in records:
            lines.append(",".join(str(row.get(h, "")) for h in headers))
        return "\n".join(lines)

    @staticmethod
    def to_graphml(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '<graph edgedefault="directed">',
        ]
        for i, node in enumerate(nodes):
            nid = escape(str(node.get("id", f"n{i}")))
            label = escape(str(node.get("label", "Node")))
            lines.append(f'<node id="{nid}"><data key="label">{label}</data></node>')
        for i, edge in enumerate(edges):
            src = escape(str(edge.get("source", "s")))
            tgt = escape(str(edge.get("target", "t")))
            etype = escape(str(edge.get("type", "RELATED")))
            lines.append(f'<edge id="e{i}" source="{src}" target="{tgt}"><data key="type">{etype}</data></edge>')
        lines.extend(["</graph>", "</graphml>"])
        return "\n".join(lines)

    @staticmethod
    def to_cypher_seed(user_id: str, twin_data: dict[str, Any]) -> str:
        props = json.dumps(twin_data, default=str)
        return (
            f"// Digital Twin export for {user_id}\n"
            f"MERGE (t:DigitalTwin {{user_id: '{user_id}'}}) SET t += {props};\n"
        )

    @staticmethod
    def to_networkx(twin_data: dict[str, Any]) -> dict[str, Any]:
        user_id = twin_data.get("user_id", "user")
        nodes = [{"id": user_id, "label": "User"}]
        edges: list[dict[str, str]] = []
        for profile in ("behaviour_profile", "emotion_profile", "symptom_profile"):
            nodes.append({"id": profile, "label": profile.replace("_", " ").title()})
            edges.append({"source": user_id, "target": profile, "type": "HAS_PROFILE"})
        return {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": edges}

    @staticmethod
    def export_twin(twin_data: dict[str, Any], fmt: str) -> str | dict[str, Any]:
        if fmt == "json":
            return GraphExportEngine.to_json(twin_data)
        if fmt == "csv":
            return GraphExportEngine.to_csv([{"key": k, "value": v} for k, v in twin_data.items()])
        if fmt == "graphml":
            nx = GraphExportEngine.to_networkx(twin_data)
            return GraphExportEngine.to_graphml(nx["nodes"], nx["links"])
        if fmt == "cypher":
            return GraphExportEngine.to_cypher_seed(twin_data.get("user_id", ""), twin_data)
        if fmt == "networkx":
            return GraphExportEngine.to_networkx(twin_data)
        return GraphExportEngine.to_json(twin_data)
