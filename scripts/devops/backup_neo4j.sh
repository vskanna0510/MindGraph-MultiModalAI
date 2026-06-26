#!/usr/bin/env bash
# Backup Neo4j data directory snapshot
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$ROOT/backups/neo4j"
mkdir -p "$BACKUP_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
FILE="$BACKUP_DIR/neo4j_${STAMP}.tar.gz"

docker exec mindgraph-dev-neo4j neo4j-admin database dump neo4j --to-path=/tmp/backup.dump 2>/dev/null || true
docker cp mindgraph-dev-neo4j:/tmp/backup.dump "$BACKUP_DIR/neo4j_${STAMP}.dump" 2>/dev/null || \
  echo "Neo4j dump requires enterprise or running container; backup skipped."
find "$BACKUP_DIR" -mtime +30 -delete 2>/dev/null || true
echo "Neo4j backup attempted: $BACKUP_DIR"
