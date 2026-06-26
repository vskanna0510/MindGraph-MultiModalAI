"""Schema migration runner."""

from __future__ import annotations

import logging
from pathlib import Path

from neo4j import AsyncSession

from graph.config.loader import get_graph_config

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Apply versioned Cypher migrations in order."""

    def __init__(self, migrations_dir: Path | None = None) -> None:
        cfg = get_graph_config()
        self._dirs = [
            cfg.migrations_path,
            cfg.platform_migrations_path,
        ]
        if migrations_dir:
            self._dirs.insert(0, migrations_dir)

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for directory in self._dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.cypher")):
                if path.name not in seen:
                    files.append(path)
                    seen.add(path.name)
        return files

    async def apply_all(self, session: AsyncSession) -> list[str]:
        applied: list[str] = []
        for path in self._collect_files():
            statements = self._parse_statements(path.read_text(encoding="utf-8"))
            for stmt in statements:
                await session.run(stmt)
            applied.append(path.name)
            logger.info("Applied migration: %s (%d statements)", path.name, len(statements))
        return applied

    @staticmethod
    def _parse_statements(content: str) -> list[str]:
        lines: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            lines.append(line)
        body = "\n".join(lines)
        return [s.strip() for s in body.split(";") if s.strip()]
