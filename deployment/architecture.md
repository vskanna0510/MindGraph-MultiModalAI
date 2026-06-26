# MindGraph++ DevOps Architecture

## Environments

| Environment | Compose File | Env File |
|-------------|--------------|----------|
| Development | `docker-compose.dev.yml` | `.env` |
| Testing | `docker-compose.test.yml` | `.env` + `APP_ENV=testing` |
| Staging | `docker-compose.prod.yml` | `.env.staging` |
| Production | `docker-compose.prod.yml` | `.env.production` |

**Never mix production credentials with development.**

## Container Stack

```
nginx → backend → postgres / redis / neo4j
                → celery-worker / celery-beat
                → ml-worker (profile: ml)
```

Monitoring (optional): `docker-compose.monitoring.yml` — Prometheus, Grafana, Loki, Alertmanager.

## Commands

```bash
make docker-dev
make docker-monitoring
make migrate
make validate
./scripts/devops/backup_postgres.sh
```

## Internal Networking

Databases are on `mindgraph-internal` / `mindgraph-prod` networks. Production compose does **not** expose PostgreSQL, Redis, or Neo4j to the host.

## Health Probes

| Service | Endpoint |
|---------|----------|
| Backend | `/api/v1/health`, `/health/liveness`, `/health/readiness` |
| Nginx | `/health` |

## Backups

| Target | Schedule | Retention |
|--------|----------|-----------|
| PostgreSQL | Daily script | 30 days |
| Neo4j | Daily script | 30 days |
| Config | Weekly (manual) | 30 days |
