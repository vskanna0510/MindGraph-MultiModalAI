# Backend Workflow

## Local Development

1. `docker compose up -d`
2. `cp .env.example .env`
3. `cd backend && uvicorn main:app --reload`
4. Open `http://localhost:8000/docs`

## Adding an Endpoint

1. Define Pydantic schemas in `app/schemas/`
2. Add repository methods if persistence is required
3. Implement service in `app/services/`
4. Add controller in `app/controllers/`
5. Register route in `app/api/v1/`
6. Document in `backend/docs/api_reference.md`
7. Add tests under `backend/tests/`

## Code Quality

```bash
make format
make lint
make test
pre-commit run --all-files
```

## Definition of Done

A backend module is complete only when it compiles, is tested, documented, linted, logged, configurable, and type-safe.
