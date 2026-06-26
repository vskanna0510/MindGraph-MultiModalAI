#!/usr/bin/env python3
"""Download DAIC-WOZ dataset (license-gated).

DAIC-WOZ requires registration at https://dcapswoz.ict.usc.edu/
This script stages downloads and verifies checksums once files are placed in
datasets/downloads/daic_woz/ or uses DAIC_DOWNLOAD_URL from environment.
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
    parser = argparse.ArgumentParser(description="Download or stage DAIC-WOZ data")
    parser.add_argument("--dest", default=str(ROOT / "datasets" / "raw" / "daic_woz"))
    parser.add_argument("--downloads", default=str(ROOT / "datasets" / "downloads" / "daic_woz"))
    args = parser.parse_args()

    dest = Path(args.dest)
    downloads = Path(args.downloads)
    downloads.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("DAIC_DOWNLOAD_URL", "")
    if url:
        target = downloads / Path(url).name
        if not target.exists():
            print(f"Downloading {url} -> {target}")
            urllib.request.urlretrieve(url, target)
        shutil.copytree(downloads, dest, dirs_exist_ok=True)
    else:
        print(
            "DAIC-WOZ is license-gated. Register at https://dcapswoz.ict.usc.edu/\n"
            f"Place extracted participant folders in: {downloads}\n"
            f"Then re-run with files present or set DAIC_DOWNLOAD_URL."
        )

    if dest.exists() and any(dest.iterdir()):
        manifest = write_checksum_manifest(dest)
        print(f"Checksum manifest written: {manifest}")
    else:
        print(f"No data at {dest}. Skipping checksum.")


if __name__ == "__main__":
    main()
