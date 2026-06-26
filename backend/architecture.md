# Backend Architecture

## Layering

```
HTTP Request
  → Middleware (request ID, CORS)
  → Controllers (thin)
  → Services (business logic)
  → Repositories (persistence)
  → PostgreSQL / Neo4j / Redis
```

## Rules

| Layer | Allowed | Forbidden |
|-------|---------|-----------|
| Controllers | Route params, call services | SQL, Cypher, business rules |
| Services | Orchestration, domain rules | HTTP routing, direct SQL |
| Repositories | CRUD, queries | Business logic |
| Schemas | Validation, serialization | Side effects |

## Packages

| Package | Responsibility |
|---------|----------------|
| `app/api` | Versioned routers |
| `app/controllers` | HTTP adapters |
| `app/services` | Business logic |
| `app/repositories` | Data access |
| `app/exceptions` | Typed errors |
| `app/logging` | Structured logs |
| `app/middleware` | Cross-cutting HTTP concerns |
| `app/workers` | Celery background tasks |

## Error Handling

All errors map to the standard `ApiResponse` envelope via `register_exception_handlers`. Stack traces are logged server-side only.

## Entry Point

`backend/main.py` → `app.main:app` for Uvicorn.
