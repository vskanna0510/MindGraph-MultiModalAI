# MindGraph++ Neo4j Knowledge Graph Platform

Production-grade temporal knowledge graph layer for MindGraph++. This module is the **intelligent memory** of the platform — not a visualization database.

## Architecture

```
Flutter App → FastAPI → Knowledge Extraction → Graph Validation → Neo4j
                                                      ↓
              Graph Analytics → Embeddings → Temporal Reasoning → XAI
```

## Directory Structure

| Layer | Path | Responsibility |
|-------|------|----------------|
| Storage | `connection/` | Async Neo4j driver, transactions |
| Entity | `entities/` | Immutable entities + factories |
| Relationship | `relationships/` | Typed edges |
| Temporal | `temporal/` | Timeline windows, session distance |
| Reasoning | `services/reasoning_service.py` | Risk trend inference |
| Analytics | `services/analytics_service.py` | Clinical trends |
| Recommendation | `services/recommendation_service.py` | Interventions |
| Explainability | `services/explainability_service.py` | Reasoning paths |
| Validation | `validators/` | Schema/constraint checks |
| Persistence | `repositories/` | **All Cypher lives here** |
| Business | `services/` | Business logic (no Cypher) |
| Cache | `cache/` | TTL cache + invalidation |
| Audit | `audit/` | Append-only mutation log |
| Migration | `migration/` + `schema/migrations/` | Versioned schema |
| Sync | `synchronization/` | Offline batch queue |
| Security | `security/` | Owner-scoped access |
| Export | `export/` | JSON/CSV export (no raw media) |

## Graph Layers

1. Storage → 2. Entity → 3. Relationship → 4. Temporal → 5. Reasoning → 6. Analytics → 7. Recommendation → 8. Explainability

## Configuration

YAML configs in `configs/`:

- `neo4j.yaml`, `graph.yaml`, `temporal.yaml`, `analytics.yaml`
- `reasoning.yaml`, `recommendation.yaml`, `cache.yaml`

Environment: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

## Migrations

Naming: `YYYYMMDD_description.cypher`

- `graphs/migrations/001_schema.cypher` (base)
- `backend/graph/schema/migrations/20260626_create_user_nodes.cypher` (platform)

Applied automatically on startup when `indexes.auto_create: true`.

## API Endpoints

Registered at `/api/v1/graph/`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/statistics` | Node/edge counts |
| GET | `/users/{id}` | User node |
| GET | `/users/{id}/timeline` | Session timeline |
| GET | `/users/{id}/risk` | Risk progression |
| GET | `/users/{id}/trends` | Clinical trends |
| GET | `/users/{id}/explain` | XAI explanation |
| GET | `/users/{id}/recommendations` | Interventions |
| POST | `/users/{id}/sessions` | Upsert session |
| POST | `/predictions` | Upsert prediction |

## Repository Pattern

```python
# Services NEVER write Cypher
service = GraphService(neo4j_session)
await service.upsert_session(user_id, session_id, owner=user_id)

# Repositories own persistence
repo = UserRepository(neo4j_session)
await repo.save(entity)
```

## ML Pipeline Bridge

- `builders/session_builder.py` — ingest extraction output
- `ml_pipeline/graph/` — GNN training, embeddings (offline)
- Shared Cypher patterns aligned with `ml_pipeline/graph/queries/cypher_library.py`

## Performance Targets

| Operation | Target |
|-----------|--------|
| Insert | < 50 ms |
| Node lookup | < 10 ms |
| Timeline query | < 50 ms |
| Recommendation | < 100 ms |

## Tests

```bash
pytest backend/graph/tests -v
make graph-test
```

## Temporal Memory (MP4 Part 2)

### Entity hierarchy
`User → Assessment → Session → Observation → Prediction → Risk → Emotion/Symptom/Behaviour → Recommendation → Intervention → TemporalEvent → Model → Research`

### Append-only philosophy
- **Stable nodes** (User, Session, Assessment): `MERGE` — identity persists
- **Historical nodes** (Prediction, Risk, Emotion, Symptom, Observation): `CREATE` — never overwritten
- Every prediction becomes historical evidence for future inference

### Temporal Memory Engine
```python
from graph.temporal import TemporalMemoryEngine

engine = TemporalMemoryEngine(neo4j_session)
await engine.record_session(user_id, session_id, previous_session_id=prev)
result = await engine.append_inference(
    user_id, session_id,
    risk_probability=0.65, confidence=0.82,
    emotions=[{"emotion_name": "Sadness", "probability": 0.7}],
    symptoms=[{"symptom_name": "Low Energy", "confidence": 0.6}],
)
```

### Longitudinal queries
`graph/queries/longitudinal.py` — risk over time, emotion evolution, symptom progression, behaviour evolution, improvement/decline trends, recommendation history.

### Snapshots
Daily / weekly / monthly / experiment snapshots via `SnapshotManager` with rollback references.

### Data retention
`graph/retention/` — soft delete, hard delete (GDPR), anonymization, consent withdrawal.

### New API endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/users/{id}/memory` | Full session memory append |
| GET | `/users/{id}/longitudinal` | Full longitudinal report |
| POST | `/users/{id}/snapshots` | Capture point-in-time snapshot |

## Graph Reasoning Engine (MP4 Part 3)

### 7-layer pipeline
`graph/reasoning/pipeline.py` — retrieval → expansion → temporal → analytics → comparison → recommendations → explainability

### Engines
- `reasoning/risk_trend.py` — slope, acceleration, EMA, confidence intervals
- `reasoning/temporal_reasoning.py` — improvement, deterioration, relapse, recovery
- `reasoning/patterns.py` — persistent symptoms, escalation patterns
- `reasoning/anomaly.py` — risk spikes, mood shifts, missing sessions
- `reasoning/alerts.py` — clinical alerts as `GraphAlert` nodes
- `analytics/graph_metrics.py` — density, centrality, node importance
- `analytics/materialized_views.py` — risk/emotion timeline projections

### Reasoning API
| Method | Path | Description |
|--------|------|-------------|
| POST | `/users/{id}/reasoning` | Full 7-layer reasoning pipeline |
| GET | `/users/{id}/analytics/risk` | Risk trend metrics |
| GET | `/users/{id}/analytics/graph` | Graph density and centrality |
| GET | `/users/{id}/analytics/emotions` | Emotion stability and co-occurrence |

## Graph Intelligence Engine (MP4 Part 4)

### Digital Cognitive Twin
`graph/twin/` — evolving per-user model built from session history:

- `digital_twin.py` — `DigitalTwin`, `DigitalTwinBuilder` (behaviour, emotion, symptom, recovery, language, interaction profiles)
- `risk_evolution.py` — 7/30/90-day forecasts, recovery/relapse probability
- `evolution_score.py` — twin maturity and knowledge growth scoring
- `story.py` — non-diagnostic narrative summaries
- `attention.py` — node/edge/temporal attention heatmaps
- `snapshots.py` — daily/weekly/monthly twin snapshots with comparison
- `intelligence.py` — `GraphIntelligenceEngine` orchestrator

### Recommendation & Explainability
- `recommendation/prioritizer.py` — tiered, scored personalized recommendations (low/medium/high risk)
- `explainability/engine.py` — full explanation path (why, sessions, symptoms, emotions, behaviours)
- `export/formats.py` — JSON, CSV, GraphML, Cypher, NetworkX export
- `evaluation/research.py` — research evaluation metrics

### Twin API
| Method | Path | Description |
|--------|------|-------------|
| GET | `/users/{id}/twin` | Digital twin summary with story and attention |
| GET | `/users/{id}/twin/dashboard` | All dashboard timelines |
| GET | `/users/{id}/twin/recommendations` | Personalized recommendations with explanations |
| GET | `/users/{id}/twin/export?fmt=json` | Export twin (json, csv, graphml, cypher, networkx) |
| GET | `/users/{id}/twin/evaluation` | Research evaluation metrics |
| POST | `/users/{id}/twin/snapshots` | Capture twin snapshot |

Configuration: `configs/twin.yaml`

Raw media belongs in secure object storage (S3/MinIO). The graph stores metadata, embeddings references, predictions, and temporal relationships only.
