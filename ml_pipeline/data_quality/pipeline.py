"""Data governance pipeline orchestrator (MP2 Part 7)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_pipeline.data_quality.audit import append_audit_log, build_audit_entry
from ml_pipeline.data_quality.config import data_quality_config, data_quality_paths
from ml_pipeline.data_quality.consistency import validate_consistency
from ml_pipeline.data_quality.embeddings import validate_embeddings, write_embedding_report
from ml_pipeline.data_quality.features import validate_features
from ml_pipeline.data_quality.integrity import validate_integrity, write_integrity_report
from ml_pipeline.data_quality.leakage import detect_leakage, write_leakage_report
from ml_pipeline.data_quality.passport import build_passport, save_passport
from ml_pipeline.data_quality.quality_engine import assess_quality
from ml_pipeline.data_quality.reproducibility import config_hash
from ml_pipeline.data_quality.reports import write_reports
from ml_pipeline.data_quality.statistics import compute_statistics, validate_statistics, write_statistics_report
from ml_pipeline.data_quality.types import DataQualityResult, GateResult, GateStatus
from ml_pipeline.data_quality.visualization import generate_figures
from ml_pipeline.feature_store.config import project_root
from ml_pipeline.feature_store.index import build_session_index, load_dataset_index


class DataQualityPipeline:
    """Full governance pipeline — no training until all gates pass."""

    def __init__(self) -> None:
        self.cfg = data_quality_config()
        self.paths = data_quality_paths()

    def _load_sessions(self) -> pd.DataFrame:
        session_path = self.paths["feature_sessions"]
        if session_path.exists():
            return pd.read_csv(session_path)
        return build_session_index(load_dataset_index())

    def _load_index(self) -> pd.DataFrame:
        if self.paths["dataset_index"].exists():
            return pd.read_csv(self.paths["dataset_index"])
        return pd.DataFrame()

    def run(self, limit: int = 0) -> DataQualityResult:
        start = datetime.now(UTC).isoformat()
        result = DataQualityResult()
        sessions_df = self._load_sessions()
        index_df = self._load_index()
        if limit > 0 and not sessions_df.empty:
            sessions_df = sessions_df.head(limit)
            pids = set(sessions_df["participant_id"].astype(str))
            index_df = index_df[index_df["participant_id"].astype(str).isin(pids)] if not index_df.empty else index_df

        gates_cfg = self.cfg.get("gates", {})
        dq_cfg = self.cfg.get("data_quality", {})

        # 1 Integrity
        if gates_cfg.get("integrity", True) and not index_df.empty:
            gate = validate_integrity(index_df, self.cfg)
            result.gates.append(gate)
            write_integrity_report(gate, self.paths["reports"] / "integrity_report.md")

        # 2 Consistency
        if gates_cfg.get("metadata", True):
            gate = validate_consistency(sessions_df)
            result.gates.append(gate)

        # 3 Quality
        quality_df = pd.DataFrame()
        if gates_cfg.get("quality", True) and not sessions_df.empty:
            gate, quality_df = assess_quality(sessions_df)
            result.gates.append(gate)

        # 4 Features
        if gates_cfg.get("features", True):
            gate = validate_features(sessions_df, project_root() / "datasets/processed/feature_store")
            result.gates.append(gate)

        # 5 Embeddings
        if gates_cfg.get("embeddings", True):
            emb_roots = [
                project_root() / "datasets/processed/text/embeddings",
                project_root() / "datasets/processed/feature_store/text",
            ]
            gate = validate_embeddings(sessions_df, emb_roots)
            result.gates.append(gate)
            write_embedding_report(gate, self.paths["reports"] / "embedding_validation_report.md")

        # 6 Statistics
        stats = compute_statistics(sessions_df)
        if gates_cfg.get("statistics", True):
            gate = validate_statistics(stats)
            result.gates.append(gate)
            write_statistics_report(stats, self.paths["reports"])

        # 7 Leakage
        if gates_cfg.get("leakage", True):
            gate = detect_leakage(sessions_df, index_df)
            result.gates.append(gate)
            write_leakage_report(gate, self.paths["reports"] / "data_leakage_report.md")

        # 8 Export validation
        export_status = GateStatus.PASSED
        manifest = project_root() / "datasets/processed/feature_store/exports/training/training_manifest.csv"
        if gates_cfg.get("export", True):
            if not manifest.exists():
                export_status = GateStatus.WARNING
            result.gates.append(
                GateResult("export", export_status, [], {"manifest_exists": manifest.exists()})
            )

        # Research approval
        failed = [g for g in result.gates if g.status == GateStatus.FAILED]
        fail_on_leakage = dq_cfg.get("fail_on_leakage", True)
        fail_on_integrity = dq_cfg.get("fail_on_integrity_error", True)
        leakage_failed = any(g.gate == "leakage" and g.status == GateStatus.FAILED for g in result.gates)
        integrity_failed = any(g.gate == "integrity" and g.status == GateStatus.FAILED for g in result.gates)
        result.approved = (
            len(failed) == 0
            and not (fail_on_leakage and leakage_failed)
            and not (fail_on_integrity and integrity_failed)
        )

        # Passport
        passport = build_passport(sessions_df, result.gates, result.approved)
        save_passport(passport, self.paths["passport"])
        result.passport = passport

        # Visualizations
        figures = generate_figures(sessions_df, quality_df, self.paths["figures"])

        # Reports
        report_paths = write_reports(result, passport, stats, self.paths["reports"])
        result.report_paths = {k: str(v) for k, v in report_paths.items()}
        if figures:
            result.report_paths["figures"] = str(figures[0])

        # Audit
        audit = build_audit_entry(result, start, config_hash(), dq_cfg.get("pipeline_version", "1.0.0"))
        result.audit = audit
        append_audit_log(audit, self.paths["audit_log"])

        # CI report
        ci_path = self.paths["ci_report"]
        ci_path.parent.mkdir(parents=True, exist_ok=True)
        ci_payload = result.to_ci_dict()
        ci_payload["timestamp"] = datetime.now(UTC).isoformat()
        ci_path.write_text(json.dumps(ci_payload, indent=2), encoding="utf-8")
        result.report_paths["ci"] = str(ci_path)

        return result
