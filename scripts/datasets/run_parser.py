#!/usr/bin/env python3
"""Run dataset parsers (Master Prompt 2 Part 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.datasets.parsers import PARSER_REGISTRY, get_parser


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ dataset parser")
    parser.add_argument("--datasets", nargs="*", default=["daic_woz", "dvlog"])
    args = parser.parse_args()

    results = {}
    for name in args.datasets:
        if name not in PARSER_REGISTRY:
            print(f"Skipping unknown parser: {name}")
            continue
        print(f"Running parser: {name}")
        p = get_parser(name)
        results[name] = p.run()
        print(f"  participants: {results[name]['participant_count']}")
        print(f"  samples: {results[name]['sample_count']}")
        print(f"  passed: {results[name]['passed']}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
