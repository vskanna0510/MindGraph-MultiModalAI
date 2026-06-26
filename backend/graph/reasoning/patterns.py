"""Graph pattern detection."""

from __future__ import annotations

from collections import Counter
from typing import Any


class PatternDetector:
    """Identify repeated behaviours, persistent symptoms, escalation patterns."""

    @staticmethod
    def repeated_items(items: list[str], min_count: int = 2) -> list[dict[str, Any]]:
        counts = Counter(items)
        return [
            {"item": item, "count": count, "pattern": "repeated"}
            for item, count in counts.items()
            if count >= min_count
        ]

    @staticmethod
    def persistent_symptoms(
        symptom_records: list[dict[str, Any]],
        min_sessions: int = 3,
    ) -> list[dict[str, Any]]:
        by_symptom: dict[str, list[float]] = {}
        for rec in symptom_records:
            name = str(rec.get("symptom", rec.get("symptom_name", "")))
            if not name:
                continue
            by_symptom.setdefault(name, []).append(float(rec.get("conf", rec.get("severity", 0))))

        persistent = []
        for name, confs in by_symptom.items():
            if len(confs) >= min_sessions:
                persistent.append({
                    "symptom": name,
                    "sessions": len(confs),
                    "avg_confidence": round(sum(confs) / len(confs), 4),
                    "pattern": "persistent",
                })
        return persistent

    @staticmethod
    def escalation_pattern(scores: list[float], threshold: float = 0.05) -> bool:
        if len(scores) < 3:
            return False
        return all(scores[i + 1] - scores[i] >= threshold for i in range(len(scores) - 1))

    @staticmethod
    def recovery_pattern(scores: list[float], threshold: float = -0.05) -> bool:
        if len(scores) < 3:
            return False
        return all(scores[i + 1] - scores[i] <= threshold for i in range(len(scores) - 1))
