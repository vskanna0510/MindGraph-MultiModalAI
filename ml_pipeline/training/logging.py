"""Training logger."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrainingLogger:
    def __init__(self, log_dir: Path, backend: str = "csv") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.csv_path = self.log_dir / "training_log.csv"
        self._rows: list[dict[str, Any]] = []
        self._fields: list[str] = []

    def log(self, epoch: int, split: str, metrics: dict[str, Any]) -> None:
        row = {"epoch": epoch, "split": split, "timestamp": datetime.now(UTC).isoformat(), **metrics}
        self._rows.append(row)
        for k in row:
            if k not in self._fields:
                self._fields.append(k)

    def flush(self) -> None:
        if not self._rows:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows(self._rows)
        summary = self.log_dir / "training_summary.json"
        summary.write_text(json.dumps(self._rows, indent=2), encoding="utf-8")
