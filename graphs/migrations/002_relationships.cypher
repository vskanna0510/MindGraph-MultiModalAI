// MindGraph++ Neo4j Schema Migration 002 — Relationships & Extended Constraints
// Apply after 001_schema.cypher

CREATE CONSTRAINT embedding_id IF NOT EXISTS FOR (e:Embedding) REQUIRE e.embedding_id IS UNIQUE;
CREATE CONSTRAINT risk_score_id IF NOT EXISTS FOR (r:RiskScore) REQUIRE r.node_id IS UNIQUE;

CREATE INDEX emotion_timestamp IF NOT EXISTS FOR (e:Emotion) ON (e.timestamp);
CREATE INDEX symptom_confidence IF NOT EXISTS FOR (s:Symptom) ON (s.confidence);
CREATE INDEX feature_checksum IF NOT EXISTS FOR (f:AudioFeature) ON (f.checksum);
CREATE INDEX model_version IF NOT EXISTS FOR (m:ModelVersion) ON (m.version);

// Relationship types are created implicitly on MERGE during graph build:
// HAS_SESSION, HAS_FEATURE, PREDICTED, EXPRESSES, INDICATES,
// TEMPORALLY_PRECEDES, FOLLOWED_BY, PART_OF_DATASET, BELONGS_TO, USES_MODEL
