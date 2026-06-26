"""Full MindGraph++ multimodal architecture."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.audio.encoder import build_audio_encoder
from ml_pipeline.models.base.model import BaseModel
from ml_pipeline.models.config import model_config
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.explainability.engine import ExplainabilityEngine
from ml_pipeline.models.fusion.fusion_registry import MultimodalFusionStack
from ml_pipeline.graph.gnn.registry import build_gnn
from ml_pipeline.models.graph.gnn import GraphSAGEModule, KnowledgeGraphInjection
from ml_pipeline.models.heads.multi_task import MultiTaskHeads
from ml_pipeline.models.heads.prediction import PredictionHead, RiskHead
from ml_pipeline.models.image.encoder import build_image_encoder
from ml_pipeline.models.text.encoder import build_text_encoder
from ml_pipeline.models.types import MultimodalBatch, TensorContract
from ml_pipeline.models.visual.encoder import build_visual_encoder


class MindGraphMultimodal(BaseModel):
    """
    Complete multimodal pipeline:
    Encoders → Projection → Cross-Modal Transformer → Temporal Fusion →
    KG Injection → GNN → Prediction → Explainability → Risk
    """

    model_id = "MODEL_010"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or model_config()
        super().__init__(cfg)
        m = cfg.get("model", {})
        latent = int(m.get("latent_dim", 512))
        dropout = float(m.get("dropout", 0.1))

        self.audio_encoder = build_audio_encoder({**cfg.get("audio", {}), "latent_dim": latent})
        self.visual_encoder = build_visual_encoder({**cfg.get("visual", {}), "latent_dim": latent})
        self.text_encoder = build_text_encoder({**cfg.get("text", {}), "latent_dim": latent})
        self.image_encoder = build_image_encoder({**cfg.get("image", {}), "latent_dim": latent})

        fcfg = {**cfg.get("fusion", {}), "d_model": latent, "dropout": float(cfg.get("fusion", {}).get("dropout", dropout))}
        tcfg = cfg.get("temporal", {})
        self.fusion_stack = MultimodalFusionStack({"fusion": fcfg, "temporal": tcfg})

        gcfg = cfg.get("graph", {})
        kg_dim = 64
        try:
            from ml_pipeline.graph.config import graph_config as gc

            kg_dim = int(gc().get("pipeline", {}).get("kg_dim", 64))
        except Exception:
            pass
        self.kg_injection = KnowledgeGraphInjection(latent, kg_dim=kg_dim)
        self.gnn = build_gnn(str(gcfg.get("architecture", "graphsage")), latent, gcfg)

        hcfg = {**cfg.get("heads", {}), "dropout": dropout}
        if hcfg.get("multi_task", True):
            self.heads = MultiTaskHeads(latent, hcfg)
            self.prediction_head = self.heads.classification
            self.risk_head = self.heads.risk
        else:
            self.heads = None
            self.prediction_head = PredictionHead(latent, int(hcfg.get("hidden_dim", 256)), int(hcfg.get("num_classes", 2)), dropout)
            self.risk_head = RiskHead(latent, int(hcfg.get("hidden_dim", 256)), int(hcfg.get("risk_levels", 3)))
        self.explainability = ExplainabilityEngine(["audio", "visual", "text", "image"])
        self.modality_dropout = float(m.get("modality_dropout", 0.3))

    def _encode_modality(
        self,
        encoder: nn.Module,
        tensor: torch.Tensor | None,
        mask: torch.Tensor | None,
        presence: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(tensor.shape[0], device=tensor.device) < self.modality_dropout
            if presence is not None:
                presence = presence * (~drop).float()
        out = encoder(tensor, mask, presence)
        if isinstance(out, EncoderOutput):
            return out.embedding
        if isinstance(out, tuple):
            return out[1]
        return out

    def forward(self, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
        presence = batch.modality_presence
        modality_embs: dict[str, torch.Tensor] = {}

        audio_emb = self._encode_modality(
            self.audio_encoder, batch.audio, batch.audio_mask, presence.get("audio"), "audio"
        )
        if audio_emb is not None:
            modality_embs["audio"] = audio_emb

        visual_emb = self._encode_modality(
            self.visual_encoder, batch.visual, batch.visual_mask, presence.get("visual"), "visual"
        )
        if visual_emb is not None:
            modality_embs["visual"] = visual_emb

        text_emb = self._encode_modality(
            self.text_encoder, batch.text, batch.text_mask, presence.get("text"), "text"
        )
        if text_emb is not None:
            modality_embs["text"] = text_emb

        image_emb = self._encode_modality(
            self.image_encoder, batch.image, None, presence.get("image"), "image"
        )
        if image_emb is not None:
            modality_embs["image"] = image_emb

        pres = {k: presence.get(k, torch.ones(modality_embs[k].shape[0], device=modality_embs[k].device)) for k in modality_embs}
        quality = batch.metadata.get("modality_quality") if batch.metadata else None
        history = batch.metadata.get("session_history") if batch.metadata else None
        session_idx = batch.metadata.get("session_idx") if batch.metadata else None
        fusion_out = self.fusion_stack(modality_embs, pres, quality, history, session_idx)
        fused = self.fusion_stack.fused_tensor(fusion_out)
        fused = self.kg_injection(fused, batch.graph)
        graph_adj = batch.metadata.get("graph_adj") if batch.metadata else None
        if fused.dim() == 3 and graph_adj is not None:
            fused = self.gnn(fused, graph_adj)
            if fused.dim() == 3:
                fused = fused.mean(dim=1)
        else:
            fused = self.gnn(fused, None)

        temporal_emb = fusion_out.temporal_embedding if fusion_out.temporal_embedding is not None else None

        if self.heads is not None:
            head_out = self.heads(fused, temporal_emb, fused)
            logits = head_out["logits"]
            risk_logits = head_out["risk_logits"]
        else:
            logits = self.prediction_head(fused)
            risk_logits = self.risk_head(fused)
            head_out = {"logits": logits, "risk_logits": risk_logits, "preds": logits.argmax(dim=-1)}

        return {
            **head_out,
            "fused": fused,
            "audio_embedding": audio_emb,
            "visual_embedding": visual_emb,
            "text_embedding": text_emb,
            "image_embedding": image_emb,
            "modality_embeddings": modality_embs,
            "fusion_output": fusion_out,
            "modality_weights": fusion_out.modality_weights,
        }

    def predict_with_explanation(self, batch: MultimodalBatch) -> dict[str, Any]:
        out = self.predict(batch)
        features = {k: v for k, v in out.items() if k.endswith("_embedding") and v is not None}
        explanation = self.explainability.explain(features, out.get("preds", out["logits"]))
        out["explanation"] = explanation
        return out

    def _dummy_batch(self) -> MultimodalBatch:
        cfg = self.config
        b, t = 1, 8
        audio_d = int(cfg.get("audio", {}).get("input_dim", 768))
        visual_d = int(cfg.get("visual", {}).get("input_dim", 768))
        text_d = int(cfg.get("text", {}).get("input_dim", 768))
        image_d = int(cfg.get("image", {}).get("input_dim", 512))
        return MultimodalBatch(
            audio=torch.zeros(b, t, audio_d),
            visual=torch.zeros(b, t, visual_d),
            text=torch.zeros(b, t, text_d),
            image=torch.zeros(b, t, image_d),
            audio_mask=torch.ones(b, t),
            visual_mask=torch.ones(b, t),
            text_mask=torch.ones(b, t),
            modality_presence={
                "audio": torch.ones(b),
                "visual": torch.ones(b),
                "text": torch.ones(b),
                "image": torch.zeros(b),
            },
        )

    def tensor_contracts(self) -> list[TensorContract]:
        latent = int(self.config.get("model", {}).get("latent_dim", 512))
        return [
            TensorContract("audio_in", ("B", "T", "D"), ("B", latent), has_sequence=True, embedding_dim=latent),
            TensorContract("fused_out", ("B", latent), ("B", latent), embedding_dim=latent),
            TensorContract("logits", ("B", latent), ("B", 2)),
        ]
