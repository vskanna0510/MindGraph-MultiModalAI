# Changelog

All notable changes to MindGraph++ are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial repository foundation (Master Prompt 1 — Part 1)
- Monorepo structure: backend, flutter_app, ml_pipeline, configs, graphs
- Docker Compose stack: PostgreSQL, Neo4j, Redis, MinIO
- Environment configuration via `.env` and `configs/*.yaml`
- FastAPI application entry with health endpoints and structured logging
- Setup scripts for Windows, Linux, and macOS
- Makefile with install, run, lint, test, docker targets
- Apache 2.0 license and contribution guidelines

### Security

- `.env.example` template with no committed secrets
- Comprehensive `.gitignore` for secrets, datasets, and model weights

## [0.1.0] - 2026-06-26

### Added

- Project initialization and development environment scaffolding

[Unreleased]: https://github.com/your-org/mindgraph-plus-plus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/mindgraph-plus-plus/releases/tag/v0.1.0
