// MindGraph++ Platform Migration — 20260627_add_temporal_edges
// Temporal chain indexes for longitudinal queries

CREATE INDEX temporal_edge_timestamp IF NOT EXISTS FOR ()-[r:TEMPORALLY_PRECEDES]-() ON (r.timestamp);
CREATE INDEX followed_by_timestamp IF NOT EXISTS FOR ()-[r:FOLLOWED_BY]-() ON (r.timestamp);
