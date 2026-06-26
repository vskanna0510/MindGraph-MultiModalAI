"""Model card generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelCard:
    overview: str
    architecture: str
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    training_dataset: str
    metrics: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    hardware: str = "CUDA / CPU"
    inference_time_ms: float = 0.0
    memory_mb: float = 0.0
    export_formats: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card\n\n## Overview\n{self.overview}\n",
            f"## Architecture\n{self.architecture}\n",
            f"## Input\n{self.input_spec}\n",
            f"## Output\n{self.output_spec}\n",
            f"## Training Dataset\n{self.training_dataset}\n",
            f"## Metrics\n{self.metrics}\n",
            "## Known Limitations\n",
        ]
        for lim in self.limitations:
            lines.append(f"- {lim}\n")
        lines.extend(
            [
                f"\n## Hardware\n{self.hardware}\n",
                f"## Inference Time\n{self.inference_time_ms:.2f} ms\n",
                f"## Memory\n{self.memory_mb:.1f} MB\n",
                f"## Export Formats\n{self.export_formats}\n",
            ]
        )
        return "".join(lines)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps(self.__dict__, indent=2, default=str), encoding="utf-8")
        return path
