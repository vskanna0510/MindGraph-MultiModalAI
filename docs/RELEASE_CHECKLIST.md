# Release Checklist

Per [ENGINEERING_CONSTITUTION.md](ENGINEERING_CONSTITUTION.md) Section 20.

## Pre-Release

- [ ] All CI workflows green (`backend`, `tests`, `docker`, `security`, `flutter`)
- [ ] Test coverage ≥ 90% (or documented exception with remediation plan)
- [ ] No high/critical vulnerabilities (`bandit`, `safety`, dependency scan)
- [ ] No open critical bugs
- [ ] CHANGELOG updated with release notes
- [ ] Migration notes documented (Alembic, Neo4j Cypher)
- [ ] API version tagged in `backend/app/__init__.py` and OpenAPI
- [ ] Docker images built and smoke-tested
- [ ] `docker compose -f deployment/docker/docker-compose.prod.yml config` validates
- [ ] Backup scripts tested (`scripts/devops/backup_postgres.sh`)
- [ ] Release notes generated (GitHub Release / `CHANGELOG.md`)

## Security

- [ ] `.env.production` not committed; secrets in vault/env only
- [ ] JWT secrets rotated from development defaults
- [ ] HTTPS/TLS configured for production nginx
- [ ] Rate limiting enabled

## Research / Thesis

- [ ] Experiment manifests archived for cited results
- [ ] Figures meet 300 DPI IEEE requirement
- [ ] Reproduction command documented in release notes

## Post-Release

- [ ] Monitor Grafana dashboards for 24h
- [ ] Verify health/readiness/liveness probes
- [ ] Tag git: `vMAJOR.MINOR.PATCH`
