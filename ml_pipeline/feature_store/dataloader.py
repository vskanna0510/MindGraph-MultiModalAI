"""Sampling and batching strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_pipeline.feature_store.config import feature_store_config
from ml_pipeline.utils.reproducibility import dataloader_generator


def build_sampler(labels: list[int], strategy: str = "random", seed: int = 42) -> Any:
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ImportError:
        return None

    if strategy == "balanced" or strategy == "weighted":
        labels_arr = np.array(labels)
        valid = labels_arr[labels_arr >= 0]
        if len(valid) == 0:
            return None
        classes, counts = np.unique(valid, return_counts=True)
        weight_map = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        weights = [weight_map.get(l, 0.0) if l >= 0 else 0.0 for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return None


def build_dataloader(dataset, batch_size: int = 8, num_workers: int = 0, shuffle: bool = False) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        return None

    cfg = feature_store_config()
    seed = int(cfg.get("sampling", {}).get("seed", 42))
    strategy = cfg.get("sampling", {}).get("strategy", "random")

    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item.get("label", -1)))

    sampler = build_sampler(labels, strategy, seed) if strategy in {"balanced", "weighted"} else None
    generator = dataloader_generator(seed)

    from ml_pipeline.feature_store.collate import collate_multimodal_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
