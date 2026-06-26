"""Graph and temporal auxiliary losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConsistencyLoss(nn.Module):
    def forward(self, graph_logits: torch.Tensor, main_logits: torch.Tensor) -> torch.Tensor:
        p_g = F.softmax(graph_logits, dim=-1)
        p_m = F.softmax(main_logits.detach(), dim=-1)
        return F.kl_div(p_g.log(), p_m, reduction="batchmean")


class TemporalConsistencyLoss(nn.Module):
    def forward(self, temporal_risks: torch.Tensor, risk_score: torch.Tensor) -> torch.Tensor:
        current = temporal_risks[:, 0] if temporal_risks.dim() == 2 else temporal_risks
        return F.mse_loss(current, risk_score.detach())
