#!/usr/bin/env bash
# Initialize PostgreSQL schema via Alembic and verify connectivity
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH=.

echo "==> Running Alembic migrations"
../.venv/bin/python -m alembic upgrade head

echo "==> Verifying PostgreSQL connectivity"
../.venv/bin/python - <<'PY'
import asyncio
from app.config.settings import get_settings
from app.database.health import check_postgres

async def main() -> None:
    status = await check_postgres(get_settings())
    if status != "healthy":
        raise SystemExit(f"PostgreSQL unhealthy: {status}")
    print("PostgreSQL: healthy")

asyncio.run(main())
PY

echo "==> Database initialization complete"
