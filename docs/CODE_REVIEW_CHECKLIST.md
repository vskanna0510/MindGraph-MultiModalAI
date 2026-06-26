# Code Review Checklist

Per [ENGINEERING_CONSTITUTION.md](ENGINEERING_CONSTITUTION.md) Section 18.

## Build & Quality

- [ ] Code compiles without errors
- [ ] Lint passes (`make lint`, `flutter analyze`)
- [ ] Format applied (`make format`, `dart format`)
- [ ] Tests pass (`make test`)
- [ ] Coverage not decreased (target 90% at release)

## Architecture

- [ ] Single responsibility per module
- [ ] FastAPI: Controller → Service → Repository (no DB in controllers)
- [ ] Flutter: no business logic in widgets; no API calls in UI
- [ ] No God objects or massive files (see size limits in constitution)
- [ ] No circular dependencies

## Security & Privacy

- [ ] No secrets, credentials, or API keys in code
- [ ] Input validation on all new endpoints
- [ ] Output sanitization where user content is returned
- [ ] Audit logging for sensitive operations
- [ ] GDPR-style delete/export hooks respected

## Code Hygiene

- [ ] No `print()` in `backend/app/`, `flutter_app/lib/`, `ml_pipeline/` (except CLI `main`)
- [ ] No TODO, FIXME, or placeholder implementations
- [ ] No commented-out code or dead imports
- [ ] No duplicate logic (DRY)
- [ ] No hardcoded paths or dataset locations (use config/env)

## Documentation

- [ ] README updated for touched modules
- [ ] OpenAPI / `api_reference.md` for API changes
- [ ] Model card for new ML models

## Clinical / Ethics (if applicable)

- [ ] No diagnostic language in UI or API
- [ ] Disclaimers on risk-related surfaces

## Naming

- [ ] Python: `snake_case` files, `PascalCase` classes
- [ ] Dart: `snake_case.dart`, `PascalCase` widgets, `FeatureScreen` screens
