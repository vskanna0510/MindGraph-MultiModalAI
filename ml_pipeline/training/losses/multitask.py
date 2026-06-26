"""Multi-task loss composition."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.training.losses.auxiliary import GraphConsistencyLoss, TemporalConsistencyLoss
from ml_pipeline.training.losses.classification import build_classification_loss
from ml_pipeline.training.losses.regression import build_regression_loss


def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    valid = labels[labels >= 0]
    if valid.numel() == 0:
        return torch.ones(num_classes)
    counts = torch.bincount(valid.long(), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


class MultiTaskLoss(nn.Module):
    """
    Total = classification + regression + graph + temporal + consistency + regularization
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        lcfg = cfg.get("loss", cfg)
        weights = lcfg.get("weights", {})
        self.w_cls = float(weights.get("classification", 1.0))
        self.w_reg = float(weights.get("regression", 0.3))
        self.w_conf = float(weights.get("confidence", 0.1))
        self.w_temp = float(weights.get("temporal", 0.2))
        self.w_graph = float(weights.get("graph", 0.2))
        self.w_cons = float(weights.get("consistency", 0.1))
        self.w_regul = float(weights.get("regularization", 0.01))

        cls_name = lcfg.get("classification", "cross_entropy")
        reg_name = lcfg.get("regression", "huber")
        smoothing = float(lcfg.get("label_smoothing", 0.0))
        focal_gamma = float(lcfg.get("focal_gamma", 2.0))
        self._class_weights_mode = lcfg.get("class_weights", "auto")
        self._class_weights: torch.Tensor | None = None

        self.cls_loss = build_classification_loss(cls_name, label_smoothing=smoothing, focal_gamma=focal_gamma)
        self.reg_loss = build_regression_loss(reg_name)
        self.graph_loss = GraphConsistencyLoss()
        self.temporal_loss = TemporalConsistencyLoss()

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, model: nn.Module | None = None) -> dict[str, torch.Tensor]:
        valid = labels >= 0
        device = outputs["logits"].device
        zero = torch.tensor(0.0, device=device)

        if valid.sum() == 0:
            return {"loss": zero, "cls_loss": zero}

        y = labels[valid].long()
        logits = outputs["logits"][valid]

        if self._class_weights is None and self._class_weights_mode == "auto":
            self._class_weights = compute_class_weights(labels).to(device)
        cls_loss = self.cls_loss(logits, y)

        reg_loss = zero
        if "risk_score" in outputs:
            target = labels.float()[valid]
            reg_loss = self.reg_loss(outputs["risk_score"][valid], target)

        risk_loss = zero
        if "risk_logits" in outputs:
            risk_loss = F.cross_entropy(outputs["risk_logits"][valid], y % outputs["risk_logits"].shape[-1])

        conf_loss = zero
        if "confidence" in outputs:
            target_conf = (outputs["preds"][valid] == y).float().unsqueeze(-1).expand_as(outputs["confidence"][valid])
            conf_loss = F.binary_cross_entropy(outputs["confidence"][valid], target_conf)

        temp_loss = zero
        if "temporal_risks" in outputs and "risk_score" in outputs:
            temp_loss = self.temporal_loss(outputs["temporal_risks"][valid], outputs["risk_score"][valid])

        graph_loss = zero
        if "graph_logits" in outputs:
            graph_loss = F.cross_entropy(outputs["graph_logits"][valid], y)
            graph_loss = graph_loss + self.graph_loss(outputs["graph_logits"][valid], logits)

        consistency = temp_loss

        regul = zero
        if model is not None and self.w_regul > 0:
            regul = sum(p.pow(2).sum() for p in model.parameters()) * self.w_regul

        total = (
            self.w_cls * cls_loss
            + self.w_reg * reg_loss
            + 0.3 * risk_loss
            + self.w_conf * conf_loss
            + self.w_temp * temp_loss
            + self.w_graph * graph_loss
            + self.w_cons * consistency
            + regul
        )
        return {
            "loss": total,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "risk_loss": risk_loss,
            "conf_loss": conf_loss,
            "temp_loss": temp_loss,
            "graph_loss": graph_loss,
            "regularization": regul,
        }


# Backward-compatible wrapper
class MultimodalLoss(MultiTaskLoss):
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, risk_logits: torch.Tensor | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        outputs = {"logits": logits}
        if risk_logits is not None:
            outputs["risk_logits"] = risk_logits
            outputs["preds"] = logits.argmax(dim=-1)
        return super().forward(outputs, labels)
