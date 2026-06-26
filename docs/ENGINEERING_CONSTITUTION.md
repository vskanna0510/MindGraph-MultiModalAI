# Engineering Constitution

**Version 1.0** — This document **overrides** all prior instructions when conflicts arise.

MindGraph++ must be indistinguishable from software built by a professional engineering organization.

---

## 1. Core Principles

Every piece of code must be: **Readable, Maintainable, Extensible, Secure, Scalable, Testable, Reusable, Documented, Observable, Deterministic, Production Ready.**

If a solution violates any principle, choose another solution.

**Never optimize for writing less code. Optimize for long-term maintainability.**

---

## 2. Absolute Prohibitions

Never: hardcode secrets, API URLs, credentials, paths, or dataset locations; duplicate business logic/widgets/API calls; write SQL in controllers or Cypher in Flutter; put business logic in widgets or networking in UI; mix repository and service logic; ignore exceptions or null safety; use `print()` in **application code** (`backend/app/`, `flutter_app/lib/`, `ml_pipeline/` except CLI `main`); leave TODO/FIXME; placeholder functions; fake data; mock implementations unless requested; commit debug code, commented-out code, unused imports, dead code, cache, datasets, or weights.

**Exception:** `scripts/` CLI tools may write to stdout/stderr for operator feedback.

---

## 3. Layer Contracts

### FastAPI
`Controller → Validation → Service → Repository → Database`  
Never bypass layers. Never return ORM models — always DTOs.

### Flutter
Widgets: display UI, receive state, trigger events only.  
Max lines: Screen 350, Widget 200, Provider 250, Repository 300.

### Python
Max ~400 lines per file (services 300, repositories 300, utilities 200).

---

## 4. Data Store Responsibilities

| Store | Owns |
|-------|------|
| PostgreSQL | Users, auth, metadata, consent, preferences, notifications |
| Neo4j | Temporal KG, predictions, symptoms, sessions, risk, history |
| Redis | Cache, rate limits, temporary sessions |

Never mix responsibilities.

---

## 5. API Contract

Every endpoint: authentication (when required), validation, logging, exception handling, OpenAPI docs, examples, response codes, versioning, pagination where applicable.

---

## 6. Security & Privacy

- JWT auth, RBAC, HTTPS, AES-256, Argon2 passwords
- Mandatory input validation, output sanitization, rate limiting, brute-force protection
- Secrets only in environment variables
- Users may download, export, delete data/account/sessions/graph/embeddings/cache; withdraw consent; disable modalities

---

## 7. Observability & Audit

Collect latency, CPU, RAM, GPU, API/graph/inference/auth/db times, cache hits/misses, queue size, Docker health.

Audit: login, logout, consent, inference, prediction, profile/graph updates, delete/export requests, failed auth, security events.  
Each record: timestamp, request ID, user ID, action, IP, device, status.

---

## 8. Testing & Performance

| Requirement | Target |
|-------------|--------|
| Minimum coverage | **90%** (release gate) |
| CI gate (current) | 70% until auth/ML modules land |
| API latency | < 500 ms |
| Prediction | < 2 s |
| Neo4j | < 100 ms |
| Flutter startup | < 2 s |

Required test types: unit, integration, API, widget, repository, database, security, performance, regression, load, inference, graph.

---

## 9. Documentation

Every folder: README. Every API: OpenAPI. Every model: model card. Every experiment: report. Databases: ER diagram. Graph: schema diagram.

---

## 10. Implementation Order (Cursor)

Think → Design → Plan → Interfaces → Models → Contracts → Implement → Test → Document → Refactor → Optimize.

**Never generate code before design.**

---

## 11. Definition of Done

✓ Implemented ✓ Compiles ✓ Tested ✓ Linted ✓ Formatted ✓ Type-safe ✓ Secure ✓ Documented ✓ Configurable ✓ Logged ✓ Monitored ✓ Production-ready

Otherwise: **Incomplete**.

---

## 12. Engineering Quality Score

Architecture, Readability, Documentation, Testing, Security, Performance, Scalability, Maintainability, Reusability — each must score **≥ 9/10** before continuing.

---

See also: [CODE_REVIEW_CHECKLIST.md](CODE_REVIEW_CHECKLIST.md), [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [development_standards.md](development_standards.md).
