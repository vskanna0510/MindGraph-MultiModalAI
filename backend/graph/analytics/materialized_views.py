"""Materialized graph projections — reusable timeline summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class MaterializedView:
    view_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    view_type: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    data: dict[str, Any] = field(default_factory=dict)
    version: str = "2.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "user_id": self.user_id,
            "view_type": self.view_type,
            "created_at": self.created_at,
            "data": self.data,
            "version": self.version,
        }


class MaterializedViewStore:
    """In-memory materialized views — risk/emotion/behaviour/recommendation timelines."""

    def __init__(self) -> None:
        self._views: dict[str, list[MaterializedView]] = {}

    def store(self, view: MaterializedView) -> MaterializedView:
        self._views.setdefault(view.user_id, []).append(view)
        return view

    def get_latest(self, user_id: str, view_type: str) -> MaterializedView | None:
        views = [v for v in self._views.get(user_id, []) if v.view_type == view_type]
        return views[-1] if views else None

    def weekly_summary(self, user_id: str, risk_data: list[dict], emotion_data: list[dict]) -> MaterializedView:
        view = MaterializedView(
            user_id=user_id,
            view_type="weekly_summary",
            data={
                "risk_points": len(risk_data),
                "emotion_points": len(emotion_data),
                "avg_risk": round(
                    sum(float(r.get("risk", 0)) for r in risk_data) / max(len(risk_data), 1), 4
                ),
            },
        )
        return self.store(view)

    def risk_timeline_view(self, user_id: str, risk_data: list[dict]) -> MaterializedView:
        view = MaterializedView(user_id=user_id, view_type="risk_timeline", data={"timeline": risk_data})
        return self.store(view)

    def emotion_timeline_view(self, user_id: str, emotion_data: list[dict]) -> MaterializedView:
        view = MaterializedView(user_id=user_id, view_type="emotion_timeline", data={"timeline": emotion_data})
        return self.store(view)
