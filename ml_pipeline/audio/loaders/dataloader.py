"""PyTorch DataLoader contract for variable-length audio."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_pipeline.utils.reproducibility import dataloader_generator


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamic padding collate with attention masks."""
    max_len = max(item["audio"].shape[0] for item in batch)
    audios = []
    masks = []
    for item in batch:
        audio = item["audio"]
        pad_len = max_len - len(audio)
        padded = np.pad(audio, (0, pad_len))
        mask = np.concatenate([np.ones(len(audio)), np.zeros(pad_len)])
        audios.append(padded)
        masks.append(mask)
    return {
        "audio": np.stack(audios),
        "attention_mask": np.stack(masks),
        "participant_ids": [item["participant_id"] for item in batch],
        "sample_rate": batch[0].get("sample_rate", 16000),
    }


def build_dataloader(dataset, batch_size: int = 8, num_workers: int = 0, seed: int = 42):
    """Build PyTorch DataLoader with custom collate."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        return None

    generator = dataloader_generator(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
    )
