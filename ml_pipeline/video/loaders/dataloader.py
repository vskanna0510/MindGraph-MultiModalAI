"""Video sequence DataLoader."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_pipeline.utils.reproducibility import dataloader_generator


def collate_video_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(item["sequence"]) for item in batch)
    dim = batch[0]["sequence"].shape[-1]
    sequences, masks = [], []
    for item in batch:
        seq = item["sequence"]
        pad = max_len - len(seq)
        padded = np.pad(seq, ((0, pad), (0, 0)))
        mask = np.concatenate([np.ones(len(seq)), np.zeros(pad)])
        sequences.append(padded)
        masks.append(mask)
    return {
        "sequences": np.stack(sequences),
        "attention_mask": np.stack(masks),
        "participant_ids": [b["participant_id"] for b in batch],
    }


def build_video_dataloader(dataset, batch_size: int = 4, num_workers: int = 0, seed: int = 42):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_video_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=dataloader_generator(seed),
    )
