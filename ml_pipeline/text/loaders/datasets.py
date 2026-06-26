"""Text dataset interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ml_pipeline.text.pipeline import TextPipeline
from ml_pipeline.text.types import TextPipelineResult


class BaseTextDataset(ABC):
    """Reusable text dataset contract."""

    name: str

    def __init__(self, raw_root: Path | None = None) -> None:
        self.raw_root = raw_root
        self.pipeline = TextPipeline()

    @abstractmethod
    def discover(self) -> list[dict]:
        """Discover transcript file records."""

    def load(self, record: dict) -> TextPipelineResult:
        return self.pipeline.process_file(
            Path(record["file_path"]),
            record["participant_id"],
            record.get("session_id"),
            record.get("split"),
        )

    def preprocess(self, record: dict) -> TextPipelineResult:
        return self.load(record)

    def extract_features(self, record: dict) -> TextPipelineResult:
        return self.load(record)

    def extract_embeddings(self, record: dict) -> TextPipelineResult:
        return self.load(record)

    def cache(self, result: TextPipelineResult) -> TextPipelineResult:
        return result

    def validate(self, result: TextPipelineResult) -> bool:
        return result.validation_passed

    def export(self, result: TextPipelineResult) -> dict:
        return self.pipeline.export_all([result])


class DAICTextDataset(BaseTextDataset):
    name = "daic_woz"

    def __init__(self, raw_root: Path | None = None) -> None:
        from ml_pipeline.datasets.config import dataset_paths

        paths = dataset_paths()
        super().__init__(raw_root or paths["raw"] / "daic_woz")

    def discover(self) -> list[dict]:
        records: list[dict] = []
        seen: set[str] = set()

        for pdir in sorted(self.raw_root.glob("*_P")):
            pid = pdir.name.replace("_P", "")
            for tfile in pdir.glob("*TRANSCRIPT*.csv"):
                key = f"{pid}:{tfile}"
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {"participant_id": pid, "session_id": pid, "file_path": str(tfile), "dataset": self.name}
                )

        for tfile in sorted(self.raw_root.glob("*_TRANSCRIPT.csv")):
            pid = tfile.name.split("_")[0]
            key = f"{pid}:{tfile}"
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {"participant_id": pid, "session_id": pid, "file_path": str(tfile), "dataset": self.name}
            )
        return records


class DVLOGTextDataset(BaseTextDataset):
    """D-VLOG has no native transcripts; discover optional sidecar text if present."""

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
            for tfile in list(pdir.glob("*.txt")) + list(pdir.glob("*transcript*.csv")):
                records.append(
                    {
                        "participant_id": pdir.name,
                        "session_id": pdir.name,
                        "file_path": str(tfile),
                        "dataset": self.name,
                    }
                )
        return records
