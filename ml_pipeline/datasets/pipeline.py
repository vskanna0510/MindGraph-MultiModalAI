"""Dataset pipeline orchestrator — executes steps 1–13 in fixed order."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_pipeline.datasets.discovery import discover_all, write_dataset_index
from ml_pipeline.datasets.logging_utils import dataset_logger
from ml_pipeline.datasets.manifest import write_manifest
from ml_pipeline.datasets.metadata import run_metadata
from ml_pipeline.datasets.processors.audio import run_audio_processing
from ml_pipeline.datasets.processors.embeddings import run_embedding_generation
from ml_pipeline.datasets.processors.export import run_export
from ml_pipeline.datasets.processors.features import run_feature_extraction
from ml_pipeline.datasets.processors.normalization import run_normalization
from ml_pipeline.datasets.processors.statistics import run_statistics
from ml_pipeline.datasets.processors.transcript import run_transcript_processing
from ml_pipeline.datasets.processors.verification import run_verification
from ml_pipeline.datasets.processors.video import run_video_processing
from ml_pipeline.datasets.quality import run_quality
from ml_pipeline.datasets import registry
from ml_pipeline.datasets.splits import run_split_verification
from ml_pipeline.datasets.types import SampleRecord
from ml_pipeline.datasets.validation_pipeline import run_validation
from ml_pipeline.datasets.versioning import write_version_manifest


@dataclass
class PipelineResult:
    """Aggregated pipeline execution result."""

    records: list[SampleRecord] = field(default_factory=list)
    steps_completed: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)


class DatasetPipeline:
    """Production dataset preparation pipeline (Master Prompt 2 Part 1)."""

    def __init__(self, datasets: list[str] | None = None) -> None:
        self.datasets = datasets or registry.list_adapters()

    def run(self, skip_processing: bool = False) -> PipelineResult:
        """Execute all 13 steps in mandated order."""
        result = PipelineResult()

        # Step 1 — Discovery
        dataset_logger.info("step_1_discovery")
        result.records = discover_all(self.datasets)
        write_dataset_index(result.records)
        result.steps_completed.append("discovery")

        # Step 2 — Validation
        dataset_logger.info("step_2_validation")
        validation_report = run_validation(result.records)
        error_counts: dict[str, int] = {}
        for issue in validation_report.issues:
            if issue.severity == "error":
                error_counts[issue.sample_id] = error_counts.get(issue.sample_id, 0) + 1
        result.steps_completed.append("validation")

        # Step 3 — Metadata
        dataset_logger.info("step_3_metadata")
        run_metadata(result.records)
        for name in self.datasets:
            write_version_manifest(name)
        result.steps_completed.append("metadata")

        # Step 4 — Split verification
        dataset_logger.info("step_4_splits")
        run_split_verification(result.records)
        result.steps_completed.append("splits")

        if skip_processing:
            run_quality(result.records, error_counts)
            missing = [i.file_path for i in validation_report.issues if i.check == "existence"]
            write_manifest(result.records, missing_files=missing)
            return result

        # Step 5 — Audio
        dataset_logger.info("step_5_audio")
        run_audio_processing(result.records)
        result.steps_completed.append("audio")

        # Step 6 — Video
        dataset_logger.info("step_6_video")
        run_video_processing(result.records)
        result.steps_completed.append("video")

        # Step 7 — Transcript
        dataset_logger.info("step_7_transcript")
        run_transcript_processing(result.records)
        result.steps_completed.append("transcript")

        # Step 8 — Features
        dataset_logger.info("step_8_features")
        run_feature_extraction(result.records)
        result.steps_completed.append("features")

        # Step 9 — Embeddings
        dataset_logger.info("step_9_embeddings")
        run_embedding_generation(result.records)
        result.steps_completed.append("embeddings")

        # Step 10 — Normalization
        dataset_logger.info("step_10_normalization")
        run_normalization(result.records)
        result.steps_completed.append("normalization")

        # Step 11 — Statistics
        dataset_logger.info("step_11_statistics")
        run_statistics(result.records)
        result.steps_completed.append("statistics")

        # Step 12 — Export
        dataset_logger.info("step_12_export")
        run_export(result.records)
        result.steps_completed.append("export")

        # Quality + manifest before final verification
        run_quality(result.records, error_counts)
        missing = [i.file_path for i in validation_report.issues if i.check == "existence"]
        corrupted = [i.file_path for i in validation_report.issues if i.check == "checksum"]
        write_manifest(result.records, missing_files=missing, corrupted_files=corrupted)

        # Step 13 — Verification
        dataset_logger.info("step_13_verification")
        result.verification = run_verification(result.records)
        result.steps_completed.append("verification")

        dataset_logger.info("pipeline_complete steps=%s", result.steps_completed)
        return result
