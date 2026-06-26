# Contributing to MindGraph++

Thank you for contributing to MindGraph++. This project follows production engineering standards suitable for academic publication and clinical-adjacent software.

## Getting Started

1. Fork the repository and clone locally.
2. Copy `.env.example` to `.env`.
3. Run the setup script for your OS (`scripts/setup.sh`, `setup_mac.sh`, or `setup.ps1`).
4. Create a branch from `develop`.

## Branch Strategy (Git Flow)

| Branch | Purpose |
|--------|---------|
| `main` | Production releases |
| `develop` | Integration branch |
| `feature/*` | New features |
| `bugfix/*` | Bug fixes |
| `hotfix/*` | Production hotfixes |
| `release/*` | Release preparation |
| `research/*` | Research experiments |
| `experiment/*` | Ablation runs |
| `paper/*` | Publication assets |

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(auth): add JWT refresh rotation
fix(api): correct inference response envelope
docs(readme): update installation guide
test(model): add fusion unit tests
```

## Code Standards

### Python

- Python 3.11+
- Black + Ruff + Mypy
- Type hints and docstrings on public APIs
- Max ~400 lines per file — refactor if exceeded

### Flutter

- Riverpod for state management
- Feature-first clean architecture
- Max ~350 lines per screen, ~300 per widget file
- Design tokens in `lib/core/theme/` — no hardcoded styles

### General

- SOLID, DRY, KISS
- No secrets in code
- No raw patient data in commits
- Every module needs `README.md`

## Pull Request Process

1. Ensure `make lint` and `make test` pass.
2. Update documentation for API or behavior changes.
3. Add tests for critical paths.
4. Request review from a maintainer.
5. Squash or rebase per maintainer guidance.

## Clinical and Ethical Guidelines

- Never use diagnostic language in UI or API responses.
- Include disclaimers on risk-related features.
- Document data handling in PR descriptions when touching PII flows.

## Questions

Open a GitHub issue with the `question` label or contact maintainers listed in `README.md`.
