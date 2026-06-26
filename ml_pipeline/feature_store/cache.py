"""Feature store cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class FeatureCache:
    def __init__(self, cache_root: Path, config_hash: str) -> None:
        self.cache_root = cache_root
        self.config_hash = config_hash
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, participant_id: str, session_id: str, feature_version: str, model_version: str) -> Path:
        digest = hashlib.sha256(
            f"{participant_id}:{session_id}:{feature_version}:{model_version}:{self.config_hash}".encode()
        ).hexdigest()[:20]
        return self.cache_root / f"{digest}.json"

    def get(self, participant_id: str, session_id: str, feature_version: str, model_version: str) -> dict[str, Any] | None:
        path = self._key(participant_id, session_id, feature_version, model_version)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self.misses += 1
        return None

    def set(self, participant_id: str, session_id: str, feature_version: str, model_version: str, payload: dict[str, Any]) -> Path:
        path = self._key(participant_id, session_id, feature_version, model_version)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return path

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def invalidate(self) -> int:
        count = 0
        for p in self.cache_root.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count
