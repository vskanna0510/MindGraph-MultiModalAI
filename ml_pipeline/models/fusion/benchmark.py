"""Benchmark fusion strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ml_pipeline.models.fusion.fusion_registry import FUSION_REGISTRY, build_fusion


def benchmark_all_strategies(
    d_model: int = 512,
    batch_size: int = 4,
    runs: int = 10,
    strategies: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    strategies = strategies or list(FUSION_REGISTRY.keys())
    results: dict[str, dict[str, float]] = {}
    embs = {m: torch.randn(batch_size, d_model) for m in ("audio", "visual", "text", "image")}
    pres = {m: torch.ones(batch_size) for m in embs}
    pres["image"] = torch.zeros(batch_size)

    for name in strategies:
        try:
            fusion = build_fusion(name, d_model, {})
            fusion.eval()
            stats = fusion.benchmark(runs=runs)
            ok, errs = fusion.validate()
            stats["valid"] = float(ok)
            if not ok:
                stats["errors"] = len(errs)
            param_count = sum(p.numel() for p in fusion.parameters())
            stats["params"] = float(param_count)
            stats["memory_mb"] = float(param_count * 4 / (1024 * 1024))
            results[name] = stats
        except Exception as exc:
            results[name] = {"error": str(exc), "valid": 0.0}
    return results


def save_benchmark_report(results: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
