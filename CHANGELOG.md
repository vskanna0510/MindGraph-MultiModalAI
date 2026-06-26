# Changelog

All notable changes to MindGraph++ are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Master Prompt 1 Part 3 (DevOps)

- Multi-environment Docker Compose: `dev`, `test`, `prod`, `monitoring`
- Dockerfiles: backend, celery, ml-worker, nginx, postgres, redis, neo4j, flutter_web
- Nginx reverse proxy with rate limiting, gzip, security headers
- Startup validation with fail-fast (`app/core/startup.py`)
- Health probes: `/health`, `/health/liveness`, `/health/readiness`
- Alembic migrations (`20260626_initial_audit_schema`)
- PostgreSQL `audit_events` model
- Database/redis/neo4j health checks
- Celery app stub and ML inference worker entry point
- DevOps scripts: `create_env`, `validate_environment`, `init_database`, backups
- Monitoring stack: Prometheus, Grafana, Loki, Alertmanager
- GitHub Actions: `backend`, `tests`, `docker`, `security`, `flutter`, `release`, `documentation`
- Environment configs: `configs/environments/{development,testing,staging,production}.yaml`
- Documentation generator: `scripts/devops/generate_docs.py`
- Enhanced pre-commit: mypy, safety, markdownlint

### Added — Master Prompt 1 Part 2
- `SECURITY.md` and security reporting policy
- `.pre-commit-config.yaml` (Black, Ruff, isort, Bandit, hooks)
- `configs/backend.yaml`, `configs/monitoring.yaml`
- Custom exception hierarchy (`AuthenticationError`, `ValidationError`, etc.)
- Global FastAPI exception handlers with standard API envelope
- `backend/main.py` Uvicorn entry point
- `app/logging/` centralized logging module
- Repository base contract (`app/repositories/base.py`)
- `scripts/ensure_module_docs.py` for README and module doc generation
- Flutter `lib/core/routes/` alias per directory standard
- ML pipeline folders: `augmentations/`, `embeddings/`, `metrics/`, `tests/`
- Backend `docs/api_reference.md`
- Exception unit tests (87% coverage on app package)

### Changed

- Makefile: `hooks`, `docs-gen` targets; `main:app` backend entry
- Health check reads `configs/backend.yaml` with `app.yaml` fallback

## [0.1.0] - 2026-06-26

### Added

- Project initialization and development environment scaffolding

[Unreleased]: https://github.com/your-org/mindgraph-plus-plus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/mindgraph-plus-plus/releases/tag/v0.1.0
