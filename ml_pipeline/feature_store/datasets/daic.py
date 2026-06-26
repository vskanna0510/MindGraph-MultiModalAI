"""DAIC-WOZ multimodal dataset."""

from __future__ import annotations

import pandas as pd

from ml_pipeline.feature_store.datasets.base import BaseMultimodalDataset
from ml_pipeline.feature_store.index import build_session_index, load_dataset_index


class DAICDataset(BaseMultimodalDataset):
    name = "daic_woz"

    def load_metadata(self) -> pd.DataFrame:
        df = build_session_index(load_dataset_index())
        self._index_df = df[df["dataset"] == self.name] if not df.empty else df
        if self.split:
            self._index_df = self._index_df[self._index_df["split"] == self.split]
        from ml_pipeline.feature_store.index import sessions_to_multimodal

        self._sessions = sessions_to_multimodal(self._index_df)
        return self._index_df
