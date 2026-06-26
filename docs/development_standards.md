# Development Standards

MindGraph++ follows **Master Prompt 1 — Part 2** engineering standards.

## Phase Order

Repository → Configuration → Virtual Environment → Dependencies → Code Quality → Git Hooks → Docker → Database → Logging → Auth → Backend → Flutter → ML → Knowledge Graph → Security → Testing → Deployment

**Do not skip phases.**

## Code Quality Tools

| Tool | Purpose |
|------|---------|
| Black | Python formatting |
| Ruff | Linting |
| Mypy | Static typing |
| isort | Import sorting |
| Bandit | Security linting |
| pre-commit | Git hooks |
| pytest + coverage | Testing (target 90%, current gate 70%) |

## File Size Limits

| Type | Max Lines |
|------|-----------|
| Python | 400 |
| Flutter widgets | 250 |
| Flutter screens | 350 |
| Services | 300 |
| Repositories | 300 |

## Naming

- Python: `snake_case` files, `PascalCase` classes, `UPPER_CASE` constants
- Flutter: `snake_case.dart` files, `PascalCase` widgets, `FeatureScreen` screens

## Documentation

Every folder: `README.md`  
Every major module: `architecture.md`, `workflow.md`, `limitations.md`, `future_work.md`  
Every API: `api_reference.md`  
Every ML model: `model_card.md`

Generate missing docs: `python scripts/ensure_module_docs.py`

## Security

See [SECURITY.md](../SECURITY.md). Never commit secrets or raw datasets.
