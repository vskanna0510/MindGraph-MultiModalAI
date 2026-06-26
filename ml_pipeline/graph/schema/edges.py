"""Edge type schemas."""

from __future__ import annotations

RELATIONSHIP_TYPES = (
    "HAS_SESSION",
    "HAS_FEATURE",
    "GENERATED",
    "PREDICTED",
    "SHOWS",
    "EXPRESSES",
    "INDICATES",
    "FOLLOWED_BY",
    "TEMPORALLY_PRECEDES",
    "RELATED_TO",
    "RECOMMENDED",
    "USES_MODEL",
    "BELONGS_TO",
    "PART_OF_DATASET",
)

EDGE_PROPERTIES = ("timestamp", "weight", "confidence", "version")

EDGE_SCHEMAS: dict[str, list[str]] = {rel: list(EDGE_PROPERTIES) for rel in RELATIONSHIP_TYPES}
