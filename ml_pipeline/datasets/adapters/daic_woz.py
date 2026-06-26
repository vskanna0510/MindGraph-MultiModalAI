"""DAIC-WOZ dataset adapter.

Supports both layouts:
  1. Full release: {participant}_P/participant_data/audio|video|transcript/
  2. Feature release: {participant}_P/{ID}_AUDIO.wav, {ID}_TRANSCRIPT.csv, etc.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ml_pipeline.datasets.adapters.base import BaseDatasetAdapter
from ml_pipeline.datasets.labels import normalize_label
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.types import LabelClass, LanguageCode, SampleRecord


class DaicWozAdapter(BaseDatasetAdapter):
    """Discover DAIC-WOZ participant sessions and modality files."""

    name = "daic_woz"
    version = "1.1.0"

    def expected_structure(self) -> list[str]:
        return [
            "{participant}_P/",
            "splits/",
        ]

    def discover(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        if not self.raw_root.exists():
            dataset_logger.warning("daic_raw_missing path=%s", self.raw_root)
            return records

        label_map = self._load_labels()
        split_map = self._load_splits()

        for participant_dir in sorted(self.raw_root.glob("*_P")):
            if not participant_dir.is_dir():
                continue
            participant_id = participant_dir.name.replace("_P", "")
            label = label_map.get(participant_id, LabelClass.UNKNOWN)
            split = split_map.get(participant_id)
            records.extend(self._discover_participant(participant_dir, participant_id, label, split))

        dataset_logger.info("daic_discovered count=%d", len(records))
        return records

    def _discover_participant(
        self,
        participant_dir: Path,
        participant_id: str,
        label: LabelClass,
        split: str | None,
    ) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        pdata = participant_dir / "participant_data"

        if pdata.exists():
            for modality, patterns in {
                "audio": ["*.wav"],
                "video": ["*.mp4", "*.avi"],
                "text": ["*.txt", "*.transcript"],
            }.items():
                sub = pdata / modality if (pdata / modality).exists() else pdata
                for pattern in patterns:
                    for file_path in sub.glob(pattern):
                        records.append(self._record(participant_id, modality, file_path, label, split))
            return records

        # Feature-release layout (flat files in participant folder)
        prefix = f"{participant_id}_"
        for file_path in sorted(participant_dir.iterdir()):
            if not file_path.is_file():
                continue
            name = file_path.name
            if name == f"{prefix}AUDIO.wav":
                records.append(self._record(participant_id, "audio", file_path, label, split))
            elif name == f"{prefix}TRANSCRIPT.csv":
                records.append(self._record(participant_id, "text", file_path, label, split))
            elif name.endswith(".mp4") or name.endswith(".avi"):
                records.append(self._record(participant_id, "video", file_path, label, split))

        return records

    def _record(
        self,
        participant_id: str,
        modality: str,
        file_path: Path,
        label: LabelClass,
        split: str | None,
    ) -> SampleRecord:
        return SampleRecord(
            sample_id=f"{participant_id}_{modality}_{file_path.stem}",
            participant_id=participant_id,
            dataset=self.name,
            session_id=participant_id,
            modality=modality,
            file_path=file_path,
            label=label,
            language=LanguageCode.ENGLISH,
            split=split,
        )

    def _load_labels(self) -> dict[str, LabelClass]:
        mapping: dict[str, LabelClass] = {}
        for labels_file in self.raw_root.rglob("*.csv"):
            name_lower = labels_file.name.lower()
            if not any(token in name_lower for token in ("label", "train", "dev", "test", "split")):
                continue
            if labels_file.parent.name.endswith("_P"):
                continue
            try:
                with labels_file.open(encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        pid = (
                            row.get("Participant_ID")
                            or row.get("participant_ID")
                            or row.get("participant_id")
                            or row.get("participant")
                            or row.get("id")
                        )
                        raw = (
                            row.get("PHQ8_Binary")
                            or row.get("PHQ8_NoInterest")
                            or row.get("label")
                            or row.get("depression")
                        )
                        if pid and raw is not None and str(raw).strip():
                            mapping[str(pid).replace("_P", "")] = normalize_label(str(raw))
            except OSError:
                continue
        return mapping

    def _load_splits(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        candidates = list(self.raw_root.glob("splits/*.csv")) + list(self.raw_root.glob("*split*.csv"))
        for csv_path in candidates:
            split_name = "train"
            lower = csv_path.name.lower()
            if "dev" in lower or "validation" in lower:
                split_name = "dev"
            elif "test" in lower:
                split_name = "test"
            try:
                with csv_path.open(encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        pid = (
                            row.get("Participant_ID")
                            or row.get("participant_ID")
                            or row.get("participant_id")
                            or row.get("id")
                        )
                        if pid:
                            mapping[str(pid).replace("_P", "")] = split_name
            except OSError:
                continue
        return mapping
