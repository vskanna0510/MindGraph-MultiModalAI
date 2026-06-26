"""Visual dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ml_pipeline.video.pipeline import VideoPipeline
from ml_pipeline.video.types import VideoPipelineResult


class BaseVisualDataset(ABC):
    name: str

    def __init__(self, raw_root: Path | None = None) -> None:
        self.raw_root = raw_root
        self.pipeline = VideoPipeline()

    @abstractmethod
    def discover(self) -> list[dict]:
        pass

    def load(self, record: dict) -> VideoPipelineResult:
        return self.pipeline.process_file(
            Path(record["file_path"]),
            record["participant_id"],
            record.get("session_id"),
            record.get("split"),
        )

    def preprocess(self, record: dict) -> VideoPipelineResult:
        return self.load(record)

    def extract_features(self, record: dict) -> VideoPipelineResult:
        return self.load(record)

    def extract_embeddings(self, record: dict) -> VideoPipelineResult:
        return self.load(record)

    def cache(self, result: VideoPipelineResult) -> VideoPipelineResult:
        return result

    def validate(self, result: VideoPipelineResult) -> bool:
        return result.validation_passed

    def export(self, result: VideoPipelineResult) -> dict:
        from ml_pipeline.video.export.exporter import export_visual_artifacts

        return export_visual_artifacts(result)


class DAICVisualDataset(BaseVisualDataset):
    name = "daic_woz"

    def __init__(self, raw_root: Path | None = None) -> None:
        from ml_pipeline.datasets.config import dataset_paths

        super().__init__(raw_root or dataset_paths()["raw"] / "daic_woz")

    def discover(self) -> list[dict]:
        records: list[dict] = []
        for pdir in sorted(self.raw_root.glob("*_P")):
            pid = pdir.name.replace("_P", "")
            for vid in list(pdir.glob("*.mp4")) + list(pdir.glob("*.avi")):
                records.append({"participant_id": pid, "session_id": pid, "file_path": str(vid), "dataset": self.name})
        return records


class DVLOGVisualDataset(BaseVisualDataset):
    name = "dvlog"

    def __init__(self, raw_root: Path | None = None) -> None:
        from ml_pipeline.datasets.config import dataset_paths

        super().__init__(raw_root or dataset_paths()["raw"] / "dvlog")

    def discover(self) -> list[dict]:
        records: list[dict] = []
        for pdir in sorted(self.raw_root.iterdir()):
            if not pdir.is_dir() or not pdir.name.isdigit():
                continue
            for npy in pdir.glob("*visual*.npy"):
                records.append(
                    {"participant_id": pdir.name, "session_id": pdir.name, "file_path": str(npy), "dataset": self.name}
                )
        return records
