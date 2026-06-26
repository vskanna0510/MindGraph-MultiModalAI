"""Graph similarity — research mode only for cross-user."""

from __future__ import annotations

import math
from typing import Any


class GraphSimilarityEngine:
    """Compute session/symptom/emotion similarity. Cross-user blocked in production."""

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return round(dot / (norm_a * norm_b), 4)

    @staticmethod
    def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return round(len(set_a & set_b) / len(union), 4)

    @staticmethod
    def session_similarity(
        symptoms_a: list[str],
        symptoms_b: list[str],
        emotions_a: list[str],
        emotions_b: list[str],
    ) -> dict[str, Any]:
        sym_sim = GraphSimilarityEngine.jaccard_similarity(set(symptoms_a), set(symptoms_b))
        emo_sim = GraphSimilarityEngine.jaccard_similarity(set(emotions_a), set(emotions_b))
        combined = (sym_sim + emo_sim) / 2
        return {
            "symptom_similarity": sym_sim,
            "emotion_similarity": emo_sim,
            "combined": round(combined, 4),
        }

    @staticmethod
    def user_similarity_research(
        user_a_data: dict[str, Any],
        user_b_data: dict[str, Any],
        *,
        research_mode: bool = False,
    ) -> dict[str, Any]:
        if not research_mode:
            raise PermissionError("Cross-user similarity requires research_mode=True")
        risks_a = user_a_data.get("risk_scores", [])
        risks_b = user_b_data.get("risk_scores", [])
        if not risks_a or not risks_b:
            return {"similarity": 0.0}
        return {
            "similarity": GraphSimilarityEngine.cosine_similarity(risks_a, risks_b),
            "research_mode": True,
        }
