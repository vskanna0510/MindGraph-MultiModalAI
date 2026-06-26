# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not** open public GitHub issues for security vulnerabilities.

Report security issues privately to the maintainers via your institution's security contact or a private GitHub security advisory once the repository is published.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested remediation (if known)

We aim to acknowledge reports within **72 hours** and provide a remediation timeline within **14 days** for confirmed issues.

## Security Principles

MindGraph++ handles sensitive mental wellness data. The following controls are mandatory:

| Control | Implementation |
|---------|----------------|
| Authentication | JWT with short-lived access tokens |
| Transport | TLS 1.3 in production |
| Secrets | Environment variables only; never committed |
| Encryption at rest | PostgreSQL, Neo4j, local mobile storage |
| Audit logging | Auth, inference, graph mutations (no raw media) |
| Rate limiting | API gateway |
| Input validation | Pydantic schemas on all endpoints |
| Least privilege | RBAC and scoped Neo4j roles |

## Prohibited in Repository

- API keys, JWT secrets, database passwords
- Raw patient datasets (DAIC-WOZ, etc.)
- Unredacted embeddings linked to identities
- Production credentials in configuration files

## Dependency Security

- Run `bandit` and `safety` (when configured) in CI
- Pin production dependencies in `requirements-prod.txt`
- Review ML dependency updates before merging

## Clinical Data Handling

This platform is **not** a medical device. Security controls protect wellness check-in data; they do not constitute HIPAA compliance unless explicitly certified for your deployment context.

## Secure Development

1. Copy `.env.example` to `.env` — never commit `.env`
2. Run `pre-commit install` after `make install`
3. Rotate `JWT_SECRET` and `ENCRYPTION_KEY` before any public deployment
4. Enable Neo4j authentication and Redis ACLs in production
