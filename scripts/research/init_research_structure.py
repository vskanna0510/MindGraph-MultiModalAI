#!/usr/bin/env python3
"""Initialize MindGraph++ research directory structure."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RESEARCH_DIRS = [
    "research/literature",
    "research/datasets",
    "research/preprocessing",
    "research/experiments",
    "research/notebooks",
    "research/reports",
    "research/ablations",
    "research/comparisons",
    "research/benchmarks",
    "research/statistics",
    "research/visualizations",
    "research/reproducibility",
    "research/publications",
    "research/supplementary",
    "ablations/audio_only",
    "ablations/video_only",
    "ablations/text_only",
    "ablations/image_only",
    "ablations/fusion",
    "ablations/graph",
    "ablations/temporal",
    "ablations/privacy",
    "ablations/sdn",
    "benchmarking/latency",
    "benchmarking/memory",
    "benchmarking/battery",
    "benchmarking/cpu",
    "benchmarking/gpu",
    "benchmarking/network",
    "benchmarking/edge",
    "benchmarking/cloud",
    "reports/daily",
    "reports/weekly",
    "reports/experiment",
    "reports/benchmark",
    "reports/evaluation",
    "datasets/raw",
    "datasets/processed",
    "datasets/intermediate",
    "datasets/cache",
    "datasets/metadata",
    "datasets/statistics",
    "datasets/splits",
    "datasets/validation",
    "datasets/downloads",
    "models/baseline",
    "models/audio",
    "models/visual",
    "models/text",
    "models/fusion",
    "models/graph",
    "models/best",
    "models/archive",
    "models/experimental",
]

README = """# {name}

MindGraph++ research module: `{path}`.

See `research/README.md` for the reproducibility workflow.
"""


def main() -> None:
    created = 0
    for relative in RESEARCH_DIRS:
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        readme = path / "README.md"
        if not readme.exists():
            name = path.name.replace("_", " ").title()
            readme.write_text(README.format(name=name, path=relative), encoding="utf-8")
            created += 1
    print(f"Research structure ready. Created {created} README files.")


if __name__ == "__main__":
    main()
