// MindGraph++ Platform Migration — 20260630_semantic_edges
// Semantic and causal relationship property indexes

CREATE INDEX emotion_id IF NOT EXISTS FOR (e:Emotion) ON (e.emotion_id);
CREATE INDEX symptom_id IF NOT EXISTS FOR (s:Symptom) ON (s.symptom_id);
CREATE INDEX causal_edge_confidence IF NOT EXISTS FOR ()-[r:MAY_INFLUENCE]-() ON (r.confidence);
CREATE INDEX semantic_edge_weight IF NOT EXISTS FOR ()-[r:CORRELATES_WITH]-() ON (r.weight);
