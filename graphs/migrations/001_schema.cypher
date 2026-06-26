// MindGraph++ Neo4j Schema Migration 001
// Apply via Neo4j Browser or: cypher-shell -f graphs/migrations/001_schema.cypher

// Uniqueness constraints
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE;
CREATE CONSTRAINT prediction_id IF NOT EXISTS FOR (p:Prediction) REQUIRE p.prediction_id IS UNIQUE;
CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.feature_id IS UNIQUE;

// Indexes for temporal queries
CREATE INDEX session_timestamp IF NOT EXISTS FOR (s:Session) ON (s.timestamp);
CREATE INDEX prediction_created IF NOT EXISTS FOR (p:Prediction) ON (p.created_at);
CREATE INDEX risk_score IF NOT EXISTS FOR (r:Risk) ON (r.score);
