"""DAIC-WOZ production parser."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.parsers.base import BaseDatasetParser
from ml_pipeline.datasets.parsers.corruption import detect_corruption
from ml_pipeline.datasets.parsers.file_discovery import (
    classify_file,
    discover_files_in_directory,
    file_to_validation_issues,
    validate_audio_file,
    validate_transcript_file,
    validate_video_file,
)
from ml_pipeline.datasets.parsers.labels import LabelValidationError, OfficialLabelParser
from ml_pipeline.datasets.parsers.quality import score_participant
from ml_pipeline.datasets.parsers.splits import validate_splits, write_split_report
from ml_pipeline.datasets.parsers.sync import compute_sync
from ml_pipeline.datasets.parsers.transcript import export_transcript, parse_transcript
from ml_pipeline.datasets.parsers.types import (
    ParticipantMetadata,
    ParserResult,
    TranscriptData,
    ValidationIssue,
)
from ml_pipeline.datasets.types import LabelClass, LanguageCode


class DAICParser(BaseDatasetParser):
    """Production-grade DAIC-WOZ parser with dynamic discovery."""

    name = "daic_woz"
    version = "1.0.0"

    def discover_participant_dirs(self) -> list[Path]:
        """Dynamically discover all participant directories."""
        if not self.raw_root.exists():
            return []
        dirs = [p for p in sorted(self.raw_root.iterdir()) if p.is_dir() and p.name.endswith("_P")]
        return dirs

    def _participant_id_from_dir(self, path: Path) -> str:
        return path.name.replace("_P", "")

    def _pick_best_file(self, files: list, modality: str) -> Path | None:
        matches = [f for f in files if f.modality == modality]
        if not matches:
            return None
        if modality == "transcript":
            preferred = [f for f in matches if "transcript" in f.path.name.lower()]
            if preferred:
                return preferred[0].path
        return matches[0].path

    def discover(self) -> ParserResult:
        """Discover participants, sessions, and modality files."""
        label_parser = OfficialLabelParser(self.raw_root)
        try:
            split_records, _ = label_parser.parse()
        except LabelValidationError as exc:
            dataset_logger.error("label_validation_failed error=%s", exc)
            split_records = []

        label_map = {r.participant_id: r.label for r in split_records}
        split_map = {r.participant_id: r.split for r in split_records}
        gender_map = {r.participant_id: r.gender for r in split_records if r.gender}

        participants: list[ParticipantMetadata] = []
        samples: list[dict] = []
        issues: list[ValidationIssue] = []

        for participant_dir in self.discover_participant_dirs():
            pid = self._participant_id_from_dir(participant_dir)
            discovered = discover_files_in_directory(participant_dir)

            # Also check participant_data subdirectory layout
            pdata = participant_dir / "participant_data"
            if pdata.exists():
                discovered.extend(discover_files_in_directory(pdata))

            files_by_modality = {
                "audio": self._pick_best_file(discovered, "audio"),
                "video": self._pick_best_file(discovered, "video"),
                "transcript": self._pick_best_file(discovered, "transcript"),
                "facial_features": self._pick_best_file(discovered, "facial_features"),
            }

            missing = [m for m, p in files_by_modality.items() if p is None and m != "facial_features"]
            meta = ParticipantMetadata(
                participant_id=pid,
                session_id=pid,
                dataset=self.name,
                gender=gender_map.get(pid),
                label=label_map.get(pid, LabelClass.UNKNOWN),
                split=split_map.get(pid),
                language=LanguageCode.ENGLISH,
                missing_files=missing,
                files=files_by_modality,
            )

            transcript_data: TranscriptData | None = None
            if files_by_modality["audio"]:
                audio_meta, audio_errs = validate_audio_file(files_by_modality["audio"])
                meta.audio_length = audio_meta.get("duration_seconds")
                meta.sample_rate = audio_meta.get("sample_rate")
                issues.extend(
                    file_to_validation_issues(pid, files_by_modality["audio"], "audio", audio_errs)
                )

            if files_by_modality["video"]:
                video_meta, video_errs = validate_video_file(files_by_modality["video"])
                meta.video_length = video_meta.get("duration_seconds")
                meta.frame_count = video_meta.get("frame_count")
                issues.extend(
                    file_to_validation_issues(pid, files_by_modality["video"], "video", video_errs)
                )

            if files_by_modality["transcript"]:
                t_meta, t_errs = validate_transcript_file(files_by_modality["transcript"])
                meta.transcript_length = t_meta.get("char_count", 0)
                issues.extend(
                    file_to_validation_issues(pid, files_by_modality["transcript"], "transcript", t_errs)
                )

            durations = [d for d in [meta.audio_length, meta.video_length] if d]
            meta.interview_duration = max(durations) if durations else None

            offset, sync_score = compute_sync(meta, None)
            meta.sync_offset = offset
            meta.sync_score = sync_score

            scores = score_participant(
                meta,
                None,
                has_audio=files_by_modality["audio"] is not None,
                has_video=files_by_modality["video"] is not None,
                has_transcript=files_by_modality["transcript"] is not None,
            )
            meta.quality_score = scores["overall"]
            issues.extend(detect_corruption(meta))

            participants.append(meta)

            for modality, file_path in files_by_modality.items():
                if file_path is None:
                    continue
                samples.append(
                    {
                        "sample_id": f"{pid}_{modality}_{file_path.stem}",
                        "participant_id": pid,
                        "session_id": pid,
                        "dataset": self.name,
                        "modality": modality,
                        "file_path": str(file_path),
                        "label": meta.label.value,
                        "split": meta.split,
                        "language": meta.language.value,
                        "gender": meta.gender,
                        "audio_length": meta.audio_length,
                        "video_length": meta.video_length,
                        "transcript_length": meta.transcript_length,
                        "sample_rate": meta.sample_rate,
                        "frame_count": meta.frame_count,
                        "interview_duration": meta.interview_duration,
                        "quality_score": meta.quality_score,
                        "sync_score": meta.sync_score,
                        "missing_files": ",".join(meta.missing_files),
                        "processing_timestamp": datetime.now(UTC).isoformat(),
                    }
                )

        dataset_logger.info("daic_parser_discovered participants=%d samples=%d", len(participants), len(samples))
        return ParserResult(
            dataset=self.name,
            participants=participants,
            samples=samples,
            validation_issues=issues,
            split_records=split_records,
        )

    def validate(self, result: ParserResult) -> list[ValidationIssue]:
        """Validate splits, labels, and files."""
        issues = list(result.validation_issues)
        discovered_ids = {p.participant_id for p in result.participants}
        split_issues, summary = validate_splits(result.split_records, discovered_ids)
        issues.extend(split_issues)
        write_split_report(summary, self.output_root.parent / "validation" / "split_report.md")
        return issues

    def parse(self) -> ParserResult:
        """Discover, validate, and export structured transcripts."""
        result = self.discover()
        extra_issues = self.validate(result)
        result.validation_issues.extend(extra_issues)

        transcript_dir = self.output_root.parent / "processed" / "transcripts"
        for participant in result.participants:
            path = participant.files.get("transcript")
            if path and path.exists():
                try:
                    data = parse_transcript(path)
                    export_transcript(data, transcript_dir, participant.participant_id)
                except OSError:
                    continue

        self._result = result
        return result
