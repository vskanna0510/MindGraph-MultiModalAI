#!/usr/bin/env python3
"""Download D-VLOG dataset.

Place official D-VLOG release in datasets/downloads/dvlog/ or set DVLOG_DOWNLOAD_URL.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.datasets.checksum import write_checksum_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or stage D-VLOG data")
    parser.add_argument("--dest", default=str(ROOT / "datasets" / "raw" / "dvlog"))
    parser.add_argument("--downloads", default=str(ROOT / "datasets" / "downloads" / "dvlog"))
    args = parser.parse_args()

    dest = Path(args.dest)
    downloads = Path(args.downloads)
    downloads.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("DVLOG_DOWNLOAD_URL", "")
    if url:
        target = downloads / Path(url).name
        if not target.exists():
            print(f"Downloading {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        for item in downloads.iterdir():
            target_path = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target_path)
    else:
        print(
            "D-VLOG requires official release access.\n"
            f"Place labels.csv, acoustic.npy, visual.npy in: {downloads}\n"
            "Or set DVLOG_DOWNLOAD_URL environment variable."
        )
        if downloads.exists() and any(downloads.iterdir()):
            shutil.copytree(downloads, dest, dirs_exist_ok=True)

    if dest.exists() and any(dest.iterdir()):
        manifest = write_checksum_manifest(dest)
        print(f"Checksum manifest written: {manifest}")


if __name__ == "__main__":
    main()
