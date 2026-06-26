"""Feature store orchestrator — validation through export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_pipeline.feature_store.cache import FeatureCache
from ml_pipeline.feature_store.config import feature_store_config, feature_store_paths
from ml_pipeline.feature_store.datasets.combined import CombinedDataset
from ml_pipeline.feature_store.datasets.daic import DAICDataset
from ml_pipeline.feature_store.datasets.dvlog import DVLOGDataset
from ml_pipeline.feature_store.export import export_for_training
from ml_pipeline.feature_store.index import build_session_index, load_dataset_index, save_session_index
from ml_pipeline.feature_store.outlier import detect_embedding_outliers, write_outlier_report
from ml_pipeline.feature_store.quality import assess_session_quality
from ml_pipeline.feature_store.registry import FeatureRegistry
from ml_pipeline.feature_store.reproducibility import build_reproducibility_manifest, config_hash
from ml_pipeline.feature_store.session_loader import SessionLoader
from ml_pipeline.feature_store.statistics import compute_dataset_statistics, write_statistics
from ml_pipeline.feature_store.sync import assess_transcript_audio_sync, write_sync_report
from ml_pipeline.feature_store.validation import validate_session_batch, write_validation_report
from ml_pipeline.feature_store.visualization import generate_visualizations


class FeatureStorePipeline:
    """MP2 Part 6 — build index, validate, stats, viz, export."""

    def __init__(self) -> None:
        self.cfg = feature_store_config()
        self.paths = feature_store_paths()
        self.registry = FeatureRegistry()
        self.cache = FeatureCache(self.paths["cache"], config_hash())
        self.loader = SessionLoader()

    def build_index(self, dataset: str = "combined") -> Path:
        df = build_session_index(load_dataset_index())
        if dataset == "daic_woz":
            df = df[df["dataset"] == "daic_woz"]
        elif dataset == "dvlog":
            df = df[df["dataset"] == "dvlog"]
        return save_session_index(df, self.paths["metadata"] / "sessions.parquet")

    def run_validation(self, limit: int = 0) -> tuple[list[dict], Path]:
        ds = CombinedDataset()
        records = ds.discover()
        if limit > 0:
            records = records[:limit]
        results: list[dict] = []
        for rec in records:
            batch = ds.load_features(rec)
            ok, errs = validate_session_batch(batch, require_features=False)
            results.append(
                {
                    "participant_id": rec["participant_id"],
                    "passed": ok,
                    "errors": errs,
                }
            )
        report = write_validation_report(results, self.paths["quality"] / "feature_validation_report.md")
        for name in self.registry.entries:
            if all(r["passed"] for r in results[:1]) or results:
                self.registry.mark_validated(name)
        self.registry.save()
        return results, report

    def run_statistics(self) -> dict:
        ds = CombinedDataset()
        df = ds.load_metadata()
        stats = compute_dataset_statistics(df)
        write_statistics(stats, self.paths["statistics"])
        return stats

    def run_visualizations(self) -> list[Path]:
        ds = CombinedDataset()
        df = ds.load_metadata()
        return generate_visualizations(df, self.paths["statistics"] / "visualizations")

    def run_sync_analysis(self, limit: int = 50) -> Path:
        ds = CombinedDataset()
        records = ds.discover()[:limit]
        scores = []
        for rec in records:
            audio_len = float(rec.get("audio_length", 0) or 0)
            tpath = Path(rec.get("transcript_path", ""))
            score = assess_transcript_audio_sync(tpath, audio_len) if tpath else rec.get("sync_score", 0.5)
            scores.append({"participant_id": rec["participant_id"], "sync_score": score})
        return write_sync_report(scores, self.paths["quality"] / "synchronization_report.md")

    def run_outlier_detection(self, limit: int = 100) -> Path:
        ds = CombinedDataset()
        records = ds.discover()[:limit]
        embeddings = []
        ids = []
        for rec in records:
            batch = ds.load_features(rec)
            for arr in (batch.audio, batch.text, batch.visual):
                if arr is not None:
                    embeddings.append(arr)
                    ids.append(rec["participant_id"])
                    break
        method = self.cfg.get("outliers", {}).get("method", "zscore")
        threshold = float(self.cfg.get("outliers", {}).get("zscore_threshold", 3.0))
        flags = detect_embedding_outliers(embeddings, method, threshold)
        outliers = [{"participant_id": pid, "reason": "embedding_outlier"} for pid, f in zip(ids, flags) if f]
        return write_outlier_report(outliers, self.paths["quality"] / "outlier_report.md")

    def run_export(self) -> dict[str, Path]:
        ds = CombinedDataset()
        records = ds.discover()
        manifest = build_reproducibility_manifest(
            hashlib.sha256(json.dumps(records[:10], default=str).encode()).hexdigest()[:16]
        )
        return export_for_training(
            records,
            self.paths["exports"] / "training",
            manifest.__dict__,
        )

    def run_all(self, limit: int = 0) -> dict:
        index_path = self.build_index()
        validation, val_report = self.run_validation(limit=limit or 10)
        stats = self.run_statistics()
        viz = self.run_visualizations()
        sync_report = self.run_sync_analysis()
        outlier_report = self.run_outlier_detection(limit=limit or 50)
        exports = self.run_export()
        passed = sum(1 for r in validation if r["passed"])
        return {
            "index": str(index_path),
            "validation_report": str(val_report),
            "validation_passed": f"{passed}/{len(validation)}",
            "statistics": stats,
            "visualizations": [str(p) for p in viz],
            "sync_report": str(sync_report),
            "outlier_report": str(outlier_report),
            "exports": {k: str(v) for k, v in exports.items()},
            "cache_hit_rate": self.cache.hit_rate,
        }
