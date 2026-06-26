# Installation Guide

## Supported Platforms

- Windows 10/11
- Ubuntu 22.04+ / WSL2
- macOS 13+

## Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Git | Latest | Version control |
| Python | 3.11+ | Backend and ML |
| Flutter | Stable | Mobile client |
| Docker Desktop | Latest | PostgreSQL, Neo4j, Redis, MinIO |
| JDK | 21 | Android builds |
| Node.js | LTS | Tooling |
| FFmpeg | Latest | Audio/video processing |
| Graphviz | Latest | Graph visualization |

### Optional (Research)

- CUDA Toolkit — GPU training
- Mininet + Ryu — SDN simulation (Ubuntu/WSL2)

## Automated Setup

```bash
# Linux / WSL2
./scripts/setup.sh

# macOS
./scripts/setup_mac.sh

# Windows PowerShell
.\scripts\setup.ps1
```

## Manual Setup

1. Clone the repository
2. `python -m venv .venv` and activate
3. `pip install -r requirements-dev.txt`
4. `cp .env.example .env` and update secrets
5. `docker compose up -d`
6. `cd backend && uvicorn app.main:app --reload`
7. `cd flutter_app && flutter pub get && flutter run`

## Validation

```bash
make test
curl http://localhost:8000/api/v1/health
```
