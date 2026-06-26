// MindGraph++ Platform Migration — 20260628_add_symptom_nodes
// Symptom and recommendation constraints

CREATE CONSTRAINT recommendation_id IF NOT EXISTS FOR (r:Recommendation) REQUIRE r.recommendation_id IS UNIQUE;
CREATE INDEX symptom_name IF NOT EXISTS FOR (s:Symptom) ON (s.symptom);
CREATE INDEX language_session IF NOT EXISTS FOR (s:Session) ON (s.language);
