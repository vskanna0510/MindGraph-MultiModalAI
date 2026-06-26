# Node schema definitions

Node labels and property schemas are defined in:

- `graph/entities/` — immutable entity models + factories
- `ml_pipeline/graph/schema/nodes.py` — ML pipeline node catalog (22 labels)

Platform identity fields (uuid, version, checksum, etc.) are enforced on all nodes via `GraphEntity`.
