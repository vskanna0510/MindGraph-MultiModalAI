"""Text DataLoader with custom collate."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TextRecordDataset(Dataset):
    def __init__(self, records: list[dict], pipeline: Any) -> None:
        self.records = records
        self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        from pathlib import Path

        result = self.pipeline.process_file(
            Path(record["file_path"]),
            record["participant_id"],
            record.get("session_id"),
            record.get("split"),
        )
        return {
            "participant_id": result.participant_id,
            "tokens": result.tokens,
            "length": len(result.tokens),
            "sentiment": result.sentiment,
            "label": record.get("label"),
        }


def collate_text_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max((item["length"] for item in batch), default=0)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for item in batch:
        tok_len = item["length"]
        pad = [0] * max_len
        mask = [1] * tok_len + [0] * (max_len - tok_len)
        padded.append(pad)
        masks.append(mask)
    return {
        "participant_ids": [b["participant_id"] for b in batch],
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": [b.get("label") for b in batch],
        "sentiments": [b["sentiment"] for b in batch],
    }


def build_dataloader(records: list[dict], pipeline: Any, batch_size: int = 8) -> DataLoader:
    ds = TextRecordDataset(records, pipeline)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_text_batch,
        pin_memory=torch.cuda.is_available(),
    )
