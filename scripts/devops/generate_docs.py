#!/usr/bin/env python3
"""Generate documentation artifacts (folder tree, OpenAPI export)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "generated"


def write_folder_tree() -> None:
    lines: list[str] = ["# MindGraph++ Folder Tree", ""]
    for path in sorted(ROOT.rglob("*")):
        if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        if path.is_dir():
            depth = len(path.relative_to(ROOT).parts)
            lines.append(f"{'  ' * depth}- {path.name}/")
    (OUTPUT / "folder_tree.md").write_text("\n".join(lines), encoding="utf-8")


def export_openapi_stub() -> None:
    stub = {
        "openapi": "3.1.0",
        "info": {"title": "MindGraph++", "version": "0.1.0"},
        "paths": {
            "/api/v1/health": {"get": {"summary": "Health check"}},
            "/api/v1/health/liveness": {"get": {"summary": "Liveness probe"}},
            "/api/v1/health/readiness": {"get": {"summary": "Readiness probe"}},
        },
    }
    (OUTPUT / "openapi-stub.json").write_text(json.dumps(stub, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_folder_tree()
    export_openapi_stub()
    print(f"Documentation artifacts written to {OUTPUT}")


if __name__ == "__main__":
    main()
