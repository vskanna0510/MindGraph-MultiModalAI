// MindGraph++ Platform Migration — 20260629_temporal_entities
// Temporal entity constraints and indexes

CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.observation_id IS UNIQUE;
CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (r:Risk) REQUIRE r.risk_id IS UNIQUE;
CREATE CONSTRAINT behaviour_id IF NOT EXISTS FOR (b:Behaviour) REQUIRE b.behaviour_id IS UNIQUE;
CREATE CONSTRAINT assessment_id IF NOT EXISTS FOR (a:Assessment) REQUIRE a.assessment_id IS UNIQUE;
CREATE CONSTRAINT intervention_id IF NOT EXISTS FOR (i:Intervention) REQUIRE i.intervention_id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:TemporalEvent) REQUIRE e.event_id IS UNIQUE;
CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:Model) REQUIRE m.model_id IS UNIQUE;
CREATE CONSTRAINT experiment_id IF NOT EXISTS FOR (r:Research) REQUIRE r.experiment_id IS UNIQUE;

CREATE INDEX observation_timestamp IF NOT EXISTS FOR (o:Observation) ON (o.timestamp);
CREATE INDEX risk_score_v2 IF NOT EXISTS FOR (r:Risk) ON (r.risk_score);
CREATE INDEX behaviour_name IF NOT EXISTS FOR (b:Behaviour) ON (b.behaviour_name);
CREATE INDEX prediction_risk_prob IF NOT EXISTS FOR (p:Prediction) ON (p.risk_probability);
CREATE INDEX deleted_at IF NOT EXISTS FOR (n:Session) ON (n.deleted_at);
