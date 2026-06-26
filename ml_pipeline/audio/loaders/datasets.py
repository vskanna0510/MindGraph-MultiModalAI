"""Audio dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ml_pipeline.audio.pipeline import AudioPipeline
from ml_pipeline.audio.types import AudioPipelineResult


class BaseAudioDataset(ABC):
    """Reusable audio dataset contract."""

    name: str

    def __init__(self, raw_root: Path | None = None) -> None:
        self.raw_root = raw_root
        self.pipeline = AudioPipeline()

    @abstractmethod
    def discover(self) -> list[dict]:
        """Discover audio file records."""

    def load(self, record: dict) -> AudioPipelineResult:
        return self.pipeline.process_file(
            Path(record["file_path"]),
            record["participant_id"],
            record.get("session_id"),
            record.get("split"),
        )

    def preprocess(self, record: dict) -> AudioPipelineResult:
        return self.load(record)

    def extract_features(self, record: dict) -> AudioPipelineResult:
        return self.load(record)

    def extract_embeddings(self, record: dict) -> AudioPipelineResult:
        return self.load(record)

    def cache(self, result: AudioPipelineResult) -> AudioPipelineResult:
        return result

    def validate(self, result: AudioPipelineResult) -> bool:
        return result.validation_passed

    def export(self, result: AudioPipelineResult) -> dict:
        from ml_pipeline.audio.export.exporter import export_artifacts

        return export_artifacts(result)


class DAICAudioDataset(BaseAudioDataset):
    name = "daic_woz"

    def __init__(self, raw_root: Path | None = None) -> None:
        from ml_pipeline.datasets.config import dataset_paths

        paths = dataset_paths()
        super().__init__(raw_root or paths["raw"] / "daic_woz")

    def discover(self) -> list[dict]:
        records: list[dict] = []
        for pdir in sorted(self.raw_root.glob("*_P")):
            pid = pdir.name.replace("_P", "")
            for wav in pdir.glob("*.wav"):
                if "audio" in wav.name.lower():
                    records.append(
                        {"participant_id": pid, "session_id": pid, "file_path": str(wav), "dataset": self.name}
                    )
        return records


class DVLOGAudioDataset(BaseAudioDataset):
    name = "dvlog"

    def __init__(self, raw_root: Path | None = None) -> None:
        from ml_pipeline.datasets.config import dataset_paths

        paths = dataset_paths()
        super().__init__(raw_root or paths["raw"] / "dvlog")

    def discover(self) -> list[dict]:
        records: list[dict] = []
        for pdir in sorted(self.raw_root.iterdir()):
            if not pdir.is_dir() or not pdir.name.isdigit():
                continue
            for npy in pdir.glob("*acoustic*.npy"):
                records.append(
                    {
                        "participant_id": pdir.name,
                        "session_id": pdir.name,
                        "file_path": str(npy),
                        "dataset": self.name,
                        "is_npy": True,
                    }
                )
        return records
