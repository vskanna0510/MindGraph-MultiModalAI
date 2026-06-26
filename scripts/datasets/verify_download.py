#!/usr/bin/env python3
"""Verify downloaded dataset integrity against checksum manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.datasets.checksum import load_checksum_manifest, verify_checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset download checksums")
    parser.add_argument("dataset_dir", help="Path to raw dataset directory")
    parser.add_argument("--manifest", default=None, help="Path to checksums.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = Path(args.manifest) if args.manifest else dataset_dir / "checksums.json"
    manifest = load_checksum_manifest(manifest_path)

    if not manifest:
        print(f"No manifest at {manifest_path}")
        sys.exit(1)

    failed = []
    for rel_path, expected in manifest.items():
        file_path = dataset_dir / rel_path
        if not file_path.exists():
            failed.append((rel_path, "missing"))
            continue
        if not verify_checksum(file_path, expected):
            failed.append((rel_path, "checksum_mismatch"))

    if failed:
        print(f"FAILED: {len(failed)} issues")
        for path, reason in failed:
            print(f"  {path}: {reason}")
        sys.exit(1)

    print(f"OK: {len(manifest)} files verified")


if __name__ == "__main__":
    main()
