// MindGraph++ Platform Migration — 20260626_create_user_nodes
// Extends base schema with platform identity fields

CREATE INDEX user_uuid IF NOT EXISTS FOR (u:User) ON (u.uuid);
CREATE INDEX session_uuid IF NOT EXISTS FOR (s:Session) ON (s.uuid);
CREATE INDEX prediction_uuid IF NOT EXISTS FOR (p:Prediction) ON (p.uuid);
CREATE INDEX emotion_label IF NOT EXISTS FOR (e:Emotion) ON (e.emotion);
CREATE INDEX recommendation_type IF NOT EXISTS FOR (r:Recommendation) ON (r.intervention_type);
CREATE INDEX session_quality IF NOT EXISTS FOR (s:Session) ON (s.quality_score);
