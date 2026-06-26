"""D-VLOG production parser."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.labels import normalize_label
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.parsers.base import BaseDatasetParser
from ml_pipeline.datasets.parsers.quality import score_participant
from ml_pipeline.datasets.parsers.types import ParticipantMetadata, ParserResult, ValidationIssue
from ml_pipeline.datasets.types import LabelClass, LanguageCode


class DVLOGParser(BaseDatasetParser):
    """Production-grade D-VLOG parser with dynamic discovery."""

    name = "dvlog"
    version = "1.0.0"

    def discover_participant_dirs(self) -> list[Path]:
        """Discover numeric participant folders dynamically."""
        if not self.raw_root.exists():
            return []
        return sorted(
            [p for p in self.raw_root.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )

    def _load_labels(self) -> dict[str, dict]:
        labels_path = self.raw_root / "labels.csv"
        mapping: dict[str, dict] = {}
        if not labels_path.exists():
            return mapping
        with labels_path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                pid = row.get("index") or row.get("participant") or row.get("id")
                if pid is None:
                    continue
                raw = row.get("label")
                fold = row.get("fold") or row.get("split")
                split = None
                if fold:
                    split = {"valid": "dev", "validation": "dev"}.get(fold.lower(), fold.lower())
                mapping[str(pid)] = {
                    "label": normalize_label(str(raw)) if raw else LabelClass.UNKNOWN,
                    "split": split,
                    "gender": row.get("gender"),
                    "duration": float(row["duration"]) if row.get("duration") else None,
                }
        return mapping

    def discover(self) -> ParserResult:
        """Discover D-VLOG participants and modality files."""
        label_map = self._load_labels()
        participants: list[ParticipantMetadata] = []
        samples: list[dict] = []
        issues: list[ValidationIssue] = []

        for participant_dir in self.discover_participant_dirs():
            pid = participant_dir.name
            info = label_map.get(pid, {})
            acoustic = list(participant_dir.glob("*acoustic*.npy"))
            visual = list(participant_dir.glob("*visual*.npy"))

            files = {
                "audio": acoustic[0] if acoustic else None,
                "visual": visual[0] if visual else None,
            }
            missing = [m for m, p in files.items() if p is None]

            audio_len = None
            if files["audio"]:
                try:
                    arr = np.load(files["audio"])
                    audio_len = float(arr.shape[0]) if arr.ndim >= 1 else None
                except (OSError, ValueError) as exc:
                    issues.append(ValidationIssue(pid, "audio", "error", str(exc)))

            meta = ParticipantMetadata(
                participant_id=pid,
                session_id=pid,
                dataset=self.name,
                gender=info.get("gender"),
                label=info.get("label", LabelClass.UNKNOWN),
                split=info.get("split"),
                language=LanguageCode.ENGLISH,
                missing_files=missing,
                files=files,
                audio_length=audio_len,
                interview_duration=info.get("duration"),
            )

            scores = score_participant(meta, None, bool(files["audio"]), bool(files["visual"]), False)
            meta.quality_score = scores["overall"]
            participants.append(meta)

            for modality, file_path in files.items():
                if file_path is None:
                    continue
                samples.append(
                    {
                        "sample_id": f"{pid}_{modality}",
                        "participant_id": pid,
                        "session_id": pid,
                        "dataset": self.name,
                        "modality": modality,
                        "file_path": str(file_path),
                        "label": meta.label.value,
                        "split": meta.split,
                        "language": meta.language.value,
                        "gender": meta.gender,
                        "duration": meta.interview_duration,
                        "quality_score": meta.quality_score,
                        "missing_files": ",".join(missing),
                        "processing_timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        dataset_logger.info("dvlog_parser_discovered participants=%d samples=%d", len(participants), len(samples))
        return ParserResult(dataset=self.name, participants=participants, samples=samples, validation_issues=issues)

    def validate(self, result: ParserResult) -> list[ValidationIssue]:
        issues = list(result.validation_issues)
        expected = len(result.participants) * 2
        if len(result.samples) < expected:
            issues.append(
                ValidationIssue(
                    "dataset",
                    "sample_count",
                    "warning",
                    f"Expected ~{expected} samples, found {len(result.samples)}",
                )
            )
        return issues

    def parse(self) -> ParserResult:
        result = self.discover()
        result.validation_issues.extend(self.validate(result))
        self._result = result
        return result
