"""Combined multi-dataset wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_pipeline.feature_store.datasets.base import BaseMultimodalDataset
from ml_pipeline.feature_store.datasets.daic import DAICDataset
from ml_pipeline.feature_store.datasets.dvlog import DVLOGDataset


class CombinedDataset(BaseMultimodalDataset):
    name = "combined"

    def __init__(self, datasets: list[str] | None = None, split: str | None = None) -> None:
        super().__init__(split=split)
        names = datasets or ["daic_woz", "dvlog"]
        self._children: list[BaseMultimodalDataset] = []
        for n in names:
            if n == "daic_woz":
                self._children.append(DAICDataset(split=split))
            elif n == "dvlog":
                self._children.append(DVLOGDataset(split=split))

    def load_metadata(self) -> pd.DataFrame:
        frames = [child.load_metadata() for child in self._children]
        self._index_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        from ml_pipeline.feature_store.index import sessions_to_multimodal

        self._sessions = sessions_to_multimodal(self._index_df)
        return self._index_df

    def discover(self) -> list[dict]:
        return self.load_metadata().to_dict(orient="records")

    def statistics(self) -> dict[str, Any]:
        return super().statistics()
