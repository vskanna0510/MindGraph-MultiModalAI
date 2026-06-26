"""Temporal knowledge graph relationship schema definitions (v2.0)."""

from __future__ import annotations

# Structural relationships
STRUCTURAL_RELATIONSHIPS: tuple[str, ...] = (
    "HAS_ASSESSMENT",
    "HAS_SESSION",
    "GENERATED",
    "PRODUCED",
    "ESTIMATES",
    "USES",
    "IDENTIFIED",
    "GENERATED_REC",
    "FOLLOWED_BY",
    "TEMPORALLY_PRECEDES",
    "HAS_EMBEDDING",
    "HAS_FEATURE",
    "HAS_OBSERVATION",
    "PART_OF_DATASET",
)

# Semantic relationships
SEMANTIC_RELATIONSHIPS: tuple[str, ...] = (
    "CORRELATES_WITH",
    "INDICATES",
    "SUPPORTED_BY",
    "TARGETS",
    "ADDRESSES",
)

# Causal relationships (probabilistic)
CAUSAL_RELATIONSHIPS: tuple[str, ...] = (
    "MAY_INFLUENCE",
)

# Legacy (MP4 Part 1 compatibility)
LEGACY_RELATIONSHIPS: tuple[str, ...] = (
    "PREDICTED",
    "EXPRESSES",
    "RECOMMENDS",
    "HAS_RISK",
    "BELONGS_TO",
    "USES_MODEL",
    "ASSESSED_BY",
)

RELATIONSHIP_TYPES: tuple[str, ...] = (
    *STRUCTURAL_RELATIONSHIPS,
    *SEMANTIC_RELATIONSHIPS,
    *CAUSAL_RELATIONSHIPS,
    *LEGACY_RELATIONSHIPS,
)

ALLOWED_EDGES: dict[str, tuple[str, str]] = {
    "HAS_ASSESSMENT": ("User", "Assessment"),
    "HAS_SESSION": ("User", "Session"),
    "GENERATED": ("Assessment", "Session"),
    "PRODUCED": ("Session", "Prediction"),
    "ESTIMATES": ("Prediction", "Risk"),
    "USES": ("Prediction", "Embedding"),
    "IDENTIFIED": ("Prediction", "Emotion"),
    "GENERATED_REC": ("Prediction", "Recommendation"),
    "FOLLOWED_BY": ("Recommendation", "Intervention"),
    "TEMPORALLY_PRECEDES": ("Session", "Session"),
    "CORRELATES_WITH": ("Emotion", "Symptom"),
    "INDICATES": ("Behaviour", "Symptom"),
    "SUPPORTED_BY": ("Risk", "Prediction"),
    "TARGETS": ("Recommendation", "Symptom"),
    "ADDRESSES": ("Intervention", "Risk"),
    "MAY_INFLUENCE": ("Symptom", "Symptom"),
    # Legacy mappings
    "PREDICTED": ("Session", "Prediction"),
    "EXPRESSES": ("Session", "Emotion"),
    "RECOMMENDS": ("Session", "Recommendation"),
    "HAS_RISK": ("Prediction", "Risk"),
    "HAS_EMBEDDING": ("Session", "Embedding"),
    "HAS_OBSERVATION": ("Session", "Observation"),
    "HAS_FEATURE": ("Session", "AudioFeature"),
}

RELATIONSHIP_PROPERTY_SCHEMA: tuple[str, ...] = (
    "uuid",
    "created_at",
    "updated_at",
    "confidence",
    "weight",
    "reason",
    "model_version",
    "evidence_source",
)
