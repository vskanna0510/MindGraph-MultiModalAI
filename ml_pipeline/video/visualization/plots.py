"""Video visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.video.config import video_paths
from ml_pipeline.video.types import FacialFeatures, LandmarkFrame


def plot_blink_timeline(features: FacialFeatures, participant_id: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    out_dir = video_paths()["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(features.ear)
    ax.set_title(f"Eye Aspect Ratio — {participant_id}")
    path = out_dir / f"{participant_id}_blink_timeline.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_head_pose(features: FacialFeatures, participant_id: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    out_dir = video_paths()["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(features.head_pose_yaw, label="yaw")
    ax.plot(features.head_pose_pitch, label="pitch")
    ax.legend()
    ax.set_title(f"Head Pose — {participant_id}")
    path = out_dir / f"{participant_id}_head_pose.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
