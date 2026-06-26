"""Learnable modality tokens."""

from __future__ import annotations

import torch
import torch.nn as nn

TOKEN_MAP = {
    "CLS": 0,
    "AUDIO": 1,
    "VISUAL": 2,
    "TEXT": 3,
    "IMAGE": 4,
    "GRAPH": 5,
    "TIME": 6,
}


class ModalityTokens(nn.Module):
    def __init__(self, d_model: int, tokens: list[str] | None = None) -> None:
        super().__init__()
        tokens = tokens or list(TOKEN_MAP.keys())
        self.token_names = tokens
        self.embed = nn.Embedding(len(tokens), d_model)
        self.name_to_id = {t: i for i, t in enumerate(tokens)}

    def get(self, name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = self.name_to_id.get(name.upper(), 0)
        return self.embed(torch.tensor([idx], device=device)).expand(batch_size, -1)

    def add_type_embedding(self, stacked: torch.Tensor, type_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(type_ids, device=stacked.device)
        return stacked + self.embed(ids).unsqueeze(0)
