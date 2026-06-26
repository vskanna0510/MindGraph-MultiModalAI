"""Multimodal collate with missing modality masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pad_1d(arrays: list[np.ndarray | None], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    batch_size = len(arrays)
    padded = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        flat = arr.ravel()
        length = min(len(flat), max_len)
        padded[i, :length] = flat[:length]
        mask[i, :length] = 1.0
    return padded, mask


def _pad_2d(arrays: list[np.ndarray | None]) -> tuple[np.ndarray | None, np.ndarray | None]:
    valid = [a for a in arrays if a is not None]
    if not valid:
        return None, None
    if all(a.ndim == 1 for a in valid):
        max_len = max(a.shape[0] for a in valid)
        return _pad_1d(arrays, max_len)
    max_t = max(a.shape[0] for a in valid)
    max_d = max(a.shape[-1] for a in valid)
    batch = np.zeros((len(arrays), max_t, max_d), dtype=np.float32)
    mask = np.zeros((len(arrays), max_t), dtype=np.float32)
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        t = min(arr.shape[0], max_t)
        d = min(arr.shape[-1], max_d)
        batch[i, :t, :d] = arr[:t, :d]
        mask[i, :t] = 1.0
    return batch, mask


def collate_multimodal_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamic padding for variable-length multimodal batches."""
    audio_list = [item.get("audio") for item in batch]
    visual_list = [item.get("visual") for item in batch]
    text_list = [item.get("text") for item in batch]

    audio, audio_mask = _pad_1d(audio_list, max((a.shape[0] for a in audio_list if a is not None), default=1))
    visual, visual_mask = _pad_2d(visual_list)
    text, text_mask = _pad_1d(text_list, max((t.shape[0] for t in text_list if t is not None), default=1))

    modality_presence = {
        "audio": np.array([item.get("modality_mask", {}).get("audio", False) for item in batch]),
        "visual": np.array([item.get("modality_mask", {}).get("visual", False) for item in batch]),
        "text": np.array([item.get("modality_mask", {}).get("text", False) for item in batch]),
    }

    result: dict[str, Any] = {
        "participant_ids": [item["participant_id"] for item in batch],
        "session_ids": [item.get("session_id", item["participant_id"]) for item in batch],
        "labels": np.array([item.get("label", -1) for item in batch]),
        "modality_presence": modality_presence,
        "audio_mask": audio_mask,
        "text_mask": text_mask,
    }
    if audio is not None:
        result["audio"] = audio
    if visual is not None:
        result["visual"] = visual
        result["visual_mask"] = visual_mask
    if text is not None:
        result["text"] = text

    try:
        import torch

        for key in ("audio", "visual", "text", "labels", "audio_mask", "text_mask", "visual_mask"):
            if key in result and isinstance(result[key], np.ndarray):
                result[key] = torch.from_numpy(result[key])
        for k, v in modality_presence.items():
            result[f"{k}_presence"] = torch.from_numpy(v.astype(np.float32))
    except ImportError:
        pass
    return result
