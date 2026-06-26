"""Multi-task prediction heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.models.heads.prediction import PredictionHead, RiskHead


class RegressionHead(nn.Module):
    """Continuous risk / PHQ approximation."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConfidenceHead(nn.Module):
    """Prediction and modality reliability."""

    def __init__(self, in_dim: int, hidden_dim: int, num_outputs: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class TemporalHead(nn.Module):
    """7/30/90-day risk and trend."""

    def __init__(self, in_dim: int, hidden_dim: int, horizons: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class GraphHead(nn.Module):
    """Graph representation refinement and calibration."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.refine = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.calibrate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.refine(x)
        return {"graph_logits": self.cls(h), "graph_confidence": torch.sigmoid(self.calibrate(h)).squeeze(-1)}


class MultiTaskHeads(nn.Module):
    """Classification + regression + confidence + temporal + graph heads."""

    def __init__(self, in_dim: int, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        hidden = int(cfg.get("hidden_dim", 256))
        dropout = float(cfg.get("dropout", 0.1))
        num_classes = int(cfg.get("num_classes", 2))
        risk_levels = int(cfg.get("risk_levels", 3))

        self.classification = PredictionHead(in_dim, hidden, num_classes, dropout)
        self.regression = RegressionHead(in_dim, hidden, dropout)
        self.risk = RiskHead(in_dim, hidden, risk_levels)
        self.confidence = ConfidenceHead(in_dim, hidden, num_outputs=5, dropout=dropout)
        self.temporal = TemporalHead(in_dim, hidden, horizons=4, dropout=dropout)
        self.graph = GraphHead(in_dim, hidden, num_classes, dropout)

    def forward(
        self,
        fused: torch.Tensor,
        temporal_emb: torch.Tensor | None = None,
        graph_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        temporal_in = temporal_emb if temporal_emb is not None else fused
        graph_in = graph_emb if graph_emb is not None else fused
        combined = fused + 0.5 * temporal_in

        logits = self.classification(combined)
        risk_logits = self.risk(combined)
        risk_score = torch.sigmoid(self.regression(combined))
        confidence = self.confidence(combined)
        temporal_out = self.temporal(temporal_in)
        graph_out = self.graph(graph_in)

        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "preds": probs.argmax(dim=-1),
            "risk_logits": risk_logits,
            "risk_score": risk_score,
            "confidence": confidence,
            "temporal_risks": temporal_out,
            "graph_logits": graph_out["graph_logits"],
            "graph_confidence": graph_out["graph_confidence"],
        }
