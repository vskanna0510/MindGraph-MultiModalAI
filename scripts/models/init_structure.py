#!/usr/bin/env python3
"""Initialize model directory structure and documentation stubs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SUBDIRS = [
    "base", "audio", "visual", "text", "image", "fusion", "graph",
    "heads", "losses", "training", "evaluation", "metrics",
    "configs", "weights", "experiments", "deployment", "explainability", "tests",
]

ARCH = "# Architecture\n\nModular encoder → fusion → graph → heads pipeline.\n"
WORKFLOW = "# Workflow\n\nConfigure via configs/model/*.yaml → instantiate MindGraphMultimodal.\n"
README = "# {name}\n\nSee architecture.md and workflow.md.\n"


def main() -> None:
    base = ROOT / "ml_pipeline" / "models"
    for sub in SUBDIRS:
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        for name, content in (("README.md", README.format(name=sub)), ("architecture.md", ARCH), ("workflow.md", WORKFLOW)):
            p = d / name
            if not p.exists():
                p.write_text(content, encoding="utf-8")
    weights = base / "weights"
    weights.mkdir(exist_ok=True)
    print(f"Model structure ready at {base}")


if __name__ == "__main__":
    main()
