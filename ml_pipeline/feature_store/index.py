"""Build multimodal session index from dataset metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_pipeline.feature_store.config import feature_store_config, feature_store_paths
from ml_pipeline.feature_store.types import MultimodalSession, QualityMetrics


def load_dataset_index(path: Path | None = None) -> pd.DataFrame:
    paths = feature_store_paths()
    index_path = path or paths["dataset_index"]
    if not index_path.exists():
        return pd.DataFrame()
    return pd.read_csv(index_path)


def build_session_index(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate per-participant sessions from flat modality rows."""
    if df is None:
        df = load_dataset_index()
    if df.empty:
        return pd.DataFrame()

    group_cols = ["participant_id", "session_id", "dataset"]
    if not all(c in df.columns for c in group_cols):
        return pd.DataFrame()

    sessions: list[dict] = []
    for (pid, sid, dataset), group in df.groupby(group_cols):
        row = group.iloc[0]
        modalities = set(group["modality"].astype(str).tolist()) if "modality" in group.columns else set()
        audio_row = group[group["modality"] == "audio"] if "modality" in group.columns else group.iloc[:1]
        transcript_row = group[group["modality"].str.contains("transcript", case=False, na=False)] if "modality" in group.columns else pd.DataFrame()
        video_row = group[group["modality"].str.contains("video|visual|facial", case=False, na=False)] if "modality" in group.columns else pd.DataFrame()

        audio_path = str(audio_row.iloc[0]["file_path"]) if len(audio_row) else ""
        transcript_path = str(transcript_row.iloc[0]["file_path"]) if len(transcript_row) else ""
        video_path = str(video_row.iloc[0]["file_path"]) if len(video_row) else ""

        sessions.append(
            {
                "participant_id": str(pid),
                "session_id": str(sid),
                "dataset": str(dataset),
                "split": str(row.get("split", "unknown")),
                "label": str(row.get("label", "unknown")),
                "language": str(row.get("language", "en")),
                "has_audio": "audio" in modalities,
                "has_transcript": any("transcript" in m for m in modalities),
                "has_visual": any(m in modalities for m in ("video", "visual", "facial_features")),
                "audio_path": audio_path,
                "transcript_path": transcript_path,
                "video_path": video_path,
                "audio_length": float(row.get("audio_length", 0) or 0),
                "transcript_length": int(row.get("transcript_length", 0) or 0),
                "quality_score": float(row.get("quality_score", 0.8) or 0.8),
                "sync_score": float(row.get("sync_score", 1.0) or 1.0),
            }
        )
    return pd.DataFrame(sessions)


def sessions_to_multimodal(df: pd.DataFrame) -> list[MultimodalSession]:
    sessions: list[MultimodalSession] = []
    for _, row in df.iterrows():
        mask = {
            "audio": bool(row.get("has_audio", False)),
            "visual": bool(row.get("has_visual", False)),
            "text": bool(row.get("has_transcript", False)),
            "image": False,
            "graph": False,
        }
        quality = QualityMetrics(
            audio=float(row.get("quality_score", 0.8)),
            text=float(row.get("quality_score", 0.8)),
            video=float(row.get("quality_score", 0.8)),
            synchronization=float(row.get("sync_score", 1.0)),
            overall=float(row.get("quality_score", 0.8)),
        )
        sessions.append(
            MultimodalSession(
                participant_id=str(row["participant_id"]),
                session_id=str(row["session_id"]),
                dataset=str(row["dataset"]),
                split=str(row.get("split", "unknown")),
                language=str(row.get("language", "en")),
                label=str(row.get("label", "unknown")),
                modality_mask=mask,
                sync_score=float(row.get("sync_score", 1.0)),
                quality=quality,
                metadata={
                    "audio_path": row.get("audio_path", ""),
                    "transcript_path": row.get("transcript_path", ""),
                    "video_path": row.get("video_path", ""),
                },
            )
        )
    return sessions


def save_session_index(df: pd.DataFrame, path: Path | None = None) -> Path:
    paths = feature_store_paths()
    cfg = feature_store_config()
    out = path or paths["metadata"] / cfg.get("paths", {}).get("metadata_file", "sessions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except ImportError:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out
