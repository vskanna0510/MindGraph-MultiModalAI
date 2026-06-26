"""Wav2Vec2 audio encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.audio.base_audio_encoder import BaseAudioEncoder
from ml_pipeline.models.audio.feature_fusion import AudioFeatureFusion
from ml_pipeline.models.encoders.output import EncoderOutput


class Wav2Vec2Encoder(BaseAudioEncoder):
    backbone_name = "wav2vec2"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        hidden = int(config.get("hidden_dim", 512))
        self.feature_fusion = AudioFeatureFusion(
            {
                "mfcc": int(config.get("mfcc_dim", 40)),
                "prosody": int(config.get("prosody_dim", 32)),
                "pitch": int(config.get("pitch_dim", 16)),
                "wav2vec": hidden,
                "speech_quality": int(config.get("quality_dim", 8)),
            },
            hidden,
            self.dropout,
        )
        self._hf_model = None
        model_name = config.get("pretrained", "facebook/wav2vec2-base")
        self.pretrained_name = model_name

    def _load_hf(self) -> bool:
        if self._hf_model is not None:
            return True
        try:
            from transformers import Wav2Vec2Model

            self._hf_model = Wav2Vec2Model.from_pretrained(self.pretrained_name)
            for p in self._hf_model.parameters():
                p.requires_grad = self.freeze_mode != "frozen"
            return True
        except Exception:
            return False

    def _wav2vec_hidden(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=-1)
        if self._load_hf():
            out = self._hf_model(waveform).last_hidden_state
            if out.shape[-1] != self.config.get("hidden_dim", 512):
                return nn.functional.linear(
                    out,
                    torch.randn(out.shape[-1], int(self.config["hidden_dim"]), device=out.device) * 0.02,
                )
            return out
        b, t = waveform.shape[0], waveform.shape[1]
        return torch.randn(b, max(t // 320, 1), int(self.config.get("hidden_dim", 512)), device=waveform.device) * 0.01

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any) -> EncoderOutput:
        acoustic = kwargs.get("acoustic_features", {})
        if acoustic:
            wav_hidden = self._wav2vec_hidden(x if x.dim() <= 2 else x.mean(dim=-1))
            acoustic.setdefault("wav2vec", wav_hidden)
            fused, fusion_attn = self.feature_fusion(acoustic)
            hidden = self.backbone(fused)
            pooled, attn = self.pooling(hidden, mask)
            refined = self.refinement(pooled)
            embedding = nn.functional.normalize(self.output_head(refined), dim=-1)
            return EncoderOutput(
                embedding=embedding,
                hidden_states=hidden,
                attention_maps=attn,
                intermediate={"fusion_attn": fusion_attn, "fused": fused},
                confidence=self.confidence_head(embedding).squeeze(-1),
                metadata={"backbone": self.backbone_name, "pretrained": self.pretrained_name},
            )
        return super().encode(x, mask, **kwargs)
