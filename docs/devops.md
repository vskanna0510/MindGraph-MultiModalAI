# DevOps Guide

See [deployment/architecture.md](../deployment/architecture.md) for container topology.

## Quick Commands

```bash
make validate          # Check local toolchain
make docker-dev        # Start development stack
make docker-monitoring # Prometheus + Grafana + Loki
make migrate           # Alembic upgrade head
make hooks             # Install and run pre-commit
python scripts/devops/generate_docs.py
```

## Environment Files

| File | Use |
|------|-----|
| `.env` | Local development |
| `.env.production` | Production only (never commit) |

## CI Quality Gates

Pull requests are rejected when formatting, lint, tests, mypy, or bandit checks fail. Coverage gate: **70%** (rising to **90%** as modules are completed).

## Logs

Runtime logs: `logs/` with rotation configured in `configs/logging.yaml` (30-day retention).
