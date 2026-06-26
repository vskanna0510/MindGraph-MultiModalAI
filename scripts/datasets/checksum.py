"""Checksum utilities for dataset downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str, algorithm: str = "sha256") -> bool:
    return file_checksum(path, algorithm) == expected.lower()


def write_checksum_manifest(directory: Path, output: Path | None = None) -> Path:
    """Generate checksum manifest for all files in directory."""
    out = output or (directory / "checksums.json")
    entries = {}
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.name not in {"checksums.json", ".gitkeep"}:
            rel = str(file_path.relative_to(directory))
            entries[rel] = file_checksum(file_path)
    out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return out


def load_checksum_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
