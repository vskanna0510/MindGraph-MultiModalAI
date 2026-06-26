"""Cross-validation protocols."""

from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np


def kfold_splits(n: int, folds: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    chunks = np.array_split(indices, folds)
    splits = []
    for i in range(folds):
        val_idx = chunks[i]
        train_idx = np.concatenate([chunks[j] for j in range(folds) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def loso_splits(participant_ids: list[str]) -> list[tuple[list[int], list[int]]]:
    unique = sorted(set(participant_ids))
    splits = []
    for held in unique:
        val_idx = [i for i, p in enumerate(participant_ids) if p == held]
        train_idx = [i for i, p in enumerate(participant_ids) if p != held]
        splits.append((train_idx, val_idx))
    return splits


def run_cross_validation(trainer_factory: Callable[[np.ndarray, np.ndarray], Any], dataset_size: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    cv_cfg = config.get("cross_validation", {})
    protocol = cv_cfg.get("protocol", "kfold")
    folds = int(cv_cfg.get("folds", 5))
    seed = int(config.get("training", {}).get("seed", 42))
    results = []
    if protocol == "kfold":
        for train_idx, val_idx in kfold_splits(dataset_size, folds, seed):
            trainer = trainer_factory(train_idx, val_idx)
            results.append(trainer.fit())
    return results
