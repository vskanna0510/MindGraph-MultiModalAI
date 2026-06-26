# Backend

FastAPI microservices for MindGraph++ (SecureGraph-MultiDep).

## Responsibilities

- API gateway and versioned REST endpoints
- Authentication and authorization
- Inference orchestration
- Knowledge graph service integration
- Analytics and notifications

## Structure

See `architecture.md` for module boundaries.

## Running

```bash
cp ../.env.example ../.env
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Tests

```bash
pytest tests -v
```
