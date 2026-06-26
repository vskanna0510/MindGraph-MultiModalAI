"""Per-modality normalization with split-safe parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NormalizationParams:
    mean: np.ndarray | float = 0.0
    std: np.ndarray | float = 1.0
    method: str = "zscore"


@dataclass
class NormalizationState:
    audio: NormalizationParams | None = None
    visual: NormalizationParams | None = None
    text: NormalizationParams | None = None
    image: NormalizationParams | None = None
    graph: NormalizationParams | None = None
    fitted_splits: set[str] = field(default_factory=set)


def fit_zscore(arrays: list[np.ndarray]) -> NormalizationParams:
    stacked = np.concatenate([a.ravel() for a in arrays if a is not None])
    return NormalizationParams(mean=float(np.mean(stacked)), std=max(float(np.std(stacked)), 1e-8), method="zscore")


def apply_normalization(arr: np.ndarray | None, params: NormalizationParams | None) -> np.ndarray | None:
    if arr is None or params is None:
        return arr
    if params.method == "zscore":
        return (arr - params.mean) / params.std
    if params.method == "l2":
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.maximum(norm, 1e-8)
    return arr


def fit_normalization_on_split(batches: list, split: str = "train") -> NormalizationState:
    state = NormalizationState()
    train = [b for b in batches if getattr(b, "split", "") == split]
    audio_arrs = [b.audio for b in train if b.audio is not None]
    visual_arrs = [b.visual for b in train if b.visual is not None]
    text_arrs = [b.text for b in train if b.text is not None]
    if audio_arrs:
        state.audio = fit_zscore(audio_arrs)
    if visual_arrs:
        state.visual = fit_zscore(visual_arrs)
    if text_arrs:
        state.text = NormalizationParams(method="l2")
    state.fitted_splits.add(split)
    return state
