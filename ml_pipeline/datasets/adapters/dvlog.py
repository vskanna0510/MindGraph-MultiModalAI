"""D-VLOG dataset adapter.

Supports layouts:
  1. Root-level: labels.csv, acoustic.npy, visual.npy
  2. Per-index folders: {index}/{index}_acoustic.npy, {index}_visual.npy
"""

from __future__ import annotations

import csv
from pathlib import Path

from ml_pipeline.datasets.adapters.base import BaseDatasetAdapter
from ml_pipeline.datasets.labels import normalize_label
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.types import LabelClass, LanguageCode, SampleRecord


class DvlogAdapter(BaseDatasetAdapter):
    """Discover D-VLOG numpy feature files and labels."""

    name = "dvlog"
    version = "1.1.0"

    def expected_structure(self) -> list[str]:
        return ["labels.csv"]

    def discover(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        if not self.raw_root.exists():
            dataset_logger.warning("dvlog_raw_missing path=%s", self.raw_root)
            return records

        label_info = self._load_labels_csv(self.raw_root / "labels.csv")

        for npy_path in sorted(self.raw_root.rglob("*.npy")):
            participant_id = self._participant_id_from_path(npy_path)
            modality = "audio" if "acoustic" in npy_path.name.lower() else "visual"
            info = label_info.get(participant_id, {})
            records.append(
                SampleRecord(
                    sample_id=f"{participant_id}_{modality}",
                    participant_id=participant_id,
                    dataset=self.name,
                    session_id=participant_id,
                    modality=modality,
                    file_path=npy_path,
                    label=info.get("label", LabelClass.UNKNOWN),
                    language=LanguageCode.ENGLISH,
                    split=info.get("split"),
                )
            )
        dataset_logger.info("dvlog_discovered count=%d", len(records))
        return records

    def _participant_id_from_path(self, path: Path) -> str:
        """Extract participant index from path or filename."""
        parent_name = path.parent.name
        if parent_name.isdigit():
            return parent_name
        stem = path.stem
        for suffix in ("_acoustic", "_visual"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem.replace("acoustic_", "").replace("visual_", "")

    def _load_labels_csv(self, path: Path) -> dict[str, dict]:
        mapping: dict[str, dict] = {}
        if not path.exists():
            return mapping
        with path.open(encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pid = row.get("index") or row.get("participant") or row.get("id") or row.get("video_id")
                raw = row.get("label") or row.get("depression")
                if pid is None:
                    continue
                fold = row.get("fold") or row.get("split")
                split = None
                if fold:
                    fold_lower = fold.lower()
                    split = {"valid": "dev", "validation": "dev"}.get(fold_lower, fold_lower)
                mapping[str(pid)] = {
                    "label": normalize_label(str(raw)) if raw else LabelClass.UNKNOWN,
                    "split": split,
                }
        return mapping
