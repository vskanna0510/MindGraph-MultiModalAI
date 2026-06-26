# Configs

YAML configuration files for MindGraph++.

All secrets remain in `.env`. Only non-sensitive parameters belong here.

| File | Purpose |
|------|---------|
| `backend.yaml` | API server, rate limits, cache TTLs |
| `database.yaml` | Connection pool settings |
| `neo4j.yaml` | Graph schema and query limits |
| `security.yaml` | JWT, DP, audit settings |
| `logging.yaml` | Log format and retention |
| `training.yaml` | ML hyperparameters |
| `deployment.yaml` | Deploy replicas and resources |
| `flutter.yaml` | Mobile client settings |
| `monitoring.yaml` | Metrics and alert thresholds |
