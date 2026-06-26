"""Official label parser — never infer labels."""

from __future__ import annotations

import csv
from pathlib import Path

from ml_pipeline.datasets.labels import normalize_label
from ml_pipeline.datasets.parsers.types import SplitInfo, ValidationIssue
from ml_pipeline.datasets.types import LabelClass


class LabelValidationError(Exception):
    """Raised when official labels conflict across split files."""


class OfficialLabelParser:
    """Read and validate official DAIC-WOZ label CSV files."""

    SPLIT_TOKENS = ("train", "dev", "test", "split")

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root

    def discover_split_files(self) -> list[Path]:
        """Find official split CSV files at dataset root or splits/ only."""
        files: list[Path] = []
        search_dirs = [self.raw_root, self.raw_root / "splits"]
        for directory in search_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.csv")):
                lower = path.name.lower()
                if any(token in lower for token in self.SPLIT_TOKENS):
                    files.append(path)
        return files

    def _split_name(self, path: Path) -> str:
        lower = path.name.lower()
        if "dev" in lower or "validation" in lower:
            return "dev"
        if "test" in lower:
            return "test"
        return "train"

    def _participant_id(self, row: dict[str, str]) -> str | None:
        for key in ("Participant_ID", "participant_ID", "participant_id", "participant", "id"):
            if row.get(key):
                return str(row[key]).replace("_P", "").strip()
        return None

    def _raw_label(self, row: dict[str, str]) -> str | None:
        for key in ("PHQ8_Binary", "label", "depression"):
            value = row.get(key)
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        return None

    def parse(self) -> tuple[list[SplitInfo], list[ValidationIssue]]:
        """Parse official labels from split files."""
        records: list[SplitInfo] = []
        issues: list[ValidationIssue] = []
        label_by_participant: dict[str, tuple[LabelClass, str]] = {}

        for csv_path in self.discover_split_files():
            split = self._split_name(csv_path)
            with csv_path.open(encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    pid = self._participant_id(row)
                    if not pid:
                        continue
                    raw_label = self._raw_label(row)
                    label = normalize_label(raw_label) if raw_label else LabelClass.UNKNOWN
                    gender = row.get("Gender") or row.get("gender")
                    records.append(
                        SplitInfo(
                            participant_id=pid,
                            split=split,
                            label=label,
                            gender=gender,
                            source_file=csv_path.name,
                        )
                    )
                    if raw_label:
                        if pid in label_by_participant:
                            existing, source = label_by_participant[pid]
                            if existing != label:
                                raise LabelValidationError(
                                    f"Label mismatch for {pid}: {existing.value} vs {label.value} "
                                    f"({source} vs {csv_path.name})"
                                )
                        else:
                            label_by_participant[pid] = (label, csv_path.name)

        return records, issues

    def label_map(self) -> dict[str, LabelClass]:
        records, _ = self.parse()
        return {r.participant_id: r.label for r in records if r.label != LabelClass.UNKNOWN}

    def split_map(self) -> dict[str, str]:
        records, _ = self.parse()
        return {r.participant_id: r.split for r in records}
