"""Graph mutation validation (v2.0)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from graph.entities.base import GraphEntity
from graph.relationships import GraphRelationship
from graph.schema.nodes.definitions import NATURAL_KEYS, NODE_PROPERTY_SCHEMAS
from graph.schema.relationships.definitions import ALLOWED_EDGES, RELATIONSHIP_TYPES

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class GraphValidator:
    """Validate schema, constraints, temporal consistency, and graph integrity."""

    def validate_entity(self, entity: GraphEntity) -> ValidationResult:
        errors: list[str] = []
        if entity.label not in NODE_PROPERTY_SCHEMAS and entity.label not in (
            "AudioFeature", "VisualFeature", "TextFeature", "ImageFeature", "Dataset",
        ):
            errors.append(f"Unknown label: {entity.label}")

        required = set(NODE_PROPERTY_SCHEMAS.get(entity.label, ()))
        natural = NATURAL_KEYS.get(entity.label)
        if natural:
            required.add(natural)

        for key in required:
            if key not in entity.properties and key not in ("completed_at", "end_time", "storage_uri"):
                errors.append(f"Missing required property '{key}' on {entity.label}")

        if not entity.identity.uuid:
            errors.append("Missing UUID")
        elif not UUID_PATTERN.match(entity.identity.uuid):
            errors.append(f"Invalid UUID format: {entity.identity.uuid}")

        if entity.identity.confidence < 0 or entity.identity.confidence > 1:
            errors.append("Confidence must be in [0, 1]")

        ts_fields = ("timestamp", "created_at", "generated_at", "start_time", "absolute_time")
        for field_name in ts_fields:
            val = entity.properties.get(field_name)
            if val:
                err = self._validate_timestamp(val, field_name)
                if err:
                    errors.append(err)

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_relationship(self, rel: GraphRelationship) -> ValidationResult:
        errors: list[str] = []
        if rel.rel_type not in RELATIONSHIP_TYPES:
            errors.append(f"Unknown relationship: {rel.rel_type}")

        allowed = ALLOWED_EDGES.get(rel.rel_type)
        if allowed is None and rel.rel_type not in ("HAS_FEATURE",):
            errors.append(f"Undefined edge type: {rel.rel_type}")

        if rel.source_value == rel.target_value and rel.rel_type == "TEMPORALLY_PRECEDES":
            errors.append("Circular temporal reference to self")

        if rel.rel_type == "MAY_INFLUENCE" and "evidence_source" not in rel.properties:
            pass  # defaults applied in with_defaults()

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_batch(
        self,
        entities: list[GraphEntity],
        relationships: list[GraphRelationship],
    ) -> ValidationResult:
        errors: list[str] = []
        seen_keys: set[str] = set()

        for entity in entities:
            result = self.validate_entity(entity)
            errors.extend(result.errors)
            natural = NATURAL_KEYS.get(entity.label)
            if natural:
                val = entity.properties.get(natural)
                if val:
                    composite = f"{entity.label}:{natural}:{val}"
                    if composite in seen_keys:
                        errors.append(f"Duplicate entity: {composite}")
                    seen_keys.add(composite)

        for rel in relationships:
            result = self.validate_relationship(rel)
            errors.extend(result.errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_temporal_order(self, timestamps: list[str]) -> ValidationResult:
        errors: list[str] = []
        parsed: list[datetime] = []
        for ts in timestamps:
            try:
                normalized = ts.replace("Z", "+00:00")
                parsed.append(datetime.fromisoformat(normalized))
            except ValueError:
                errors.append(f"Invalid timestamp: {ts}")
        for i in range(1, len(parsed)):
            if parsed[i] < parsed[i - 1]:
                errors.append(f"Temporal ordering violation at index {i}")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_session_connectivity(
        self,
        session_ids: list[str],
        linked_pairs: list[tuple[str, str]],
    ) -> ValidationResult:
        errors: list[str] = []
        if len(session_ids) < 2:
            return ValidationResult(valid=True)
        linked = set(linked_pairs)
        for i in range(len(session_ids) - 1):
            pair = (session_ids[i], session_ids[i + 1])
            if pair not in linked and (pair[1], pair[0]) not in linked:
                errors.append(f"Disconnected session gap: {pair[0]} -> {pair[1]}")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_version_compatibility(self, entity: GraphEntity, expected_schema: str) -> ValidationResult:
        if entity.identity.schema_version != expected_schema:
            return ValidationResult(
                valid=False,
                errors=[f"Schema version mismatch: {entity.identity.schema_version} != {expected_schema}"],
            )
        return ValidationResult(valid=True)

    @staticmethod
    def _validate_timestamp(value: str, field_name: str) -> str | None:
        try:
            normalized = value.replace("Z", "+00:00")
            ts = datetime.fromisoformat(normalized)
            if ts > datetime.now(UTC):
                return f"Future timestamp in {field_name}: {value}"
        except ValueError:
            return f"Invalid timestamp in {field_name}: {value}"
        return None
