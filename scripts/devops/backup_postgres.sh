#!/usr/bin/env bash
# Backup PostgreSQL — retention managed externally (30 days recommended)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$ROOT/backups/postgres"
mkdir -p "$BACKUP_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FILE="$BACKUP_DIR/mindgraph_${STAMP}.sql.gz"

docker exec mindgraph-dev-postgres pg_dump -U "${POSTGRES_USER:-mindgraph}" "${POSTGRES_DB:-mindgraph}" | gzip > "$FILE"
echo "Backup written: $FILE"
find "$BACKUP_DIR" -name '*.sql.gz' -mtime +30 -delete
