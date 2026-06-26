#!/usr/bin/env python3
"""Resume interrupted dataset downloads."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resume_download(url: str, dest: Path) -> Path:
    """Resume HTTP download using Range header when partial file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_size = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url)
    if existing_size > 0:
        request.add_header("Range", f"bytes={existing_size}-")
    with urllib.request.urlopen(request) as response:
        mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
        with dest.open(mode) as handle:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                handle.write(chunk)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume dataset download")
    parser.add_argument("--url", default=os.environ.get("DATASET_DOWNLOAD_URL", ""))
    parser.add_argument("--dest", required=True, help="Destination file path")
    args = parser.parse_args()

    if not args.url:
        print("Set --url or DATASET_DOWNLOAD_URL")
        sys.exit(1)

    dest = resume_download(args.url, Path(args.dest))
    print(f"Download complete: {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
