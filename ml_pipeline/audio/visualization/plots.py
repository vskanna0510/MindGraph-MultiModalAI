"""Audio visualization generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.audio.config import audio_paths


def generate_audio_plots(audio: np.ndarray, sample_rate: int, participant_id: str) -> list[Path]:
    """Generate waveform and spectrogram plots."""
    out_dir = audio_paths()["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return created

    fig, ax = plt.subplots(figsize=(10, 2))
    t = np.arange(len(audio)) / sample_rate
    ax.plot(t, audio, linewidth=0.5)
    ax.set_title(f"Waveform — {participant_id}")
    wave_path = out_dir / f"{participant_id}_waveform.png"
    fig.tight_layout()
    fig.savefig(wave_path, dpi=100)
    plt.close(fig)
    created.append(wave_path)

    try:
        import librosa

        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        log_mel = librosa.power_to_db(mel)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f"Mel Spectrogram — {participant_id}")
        spec_path = out_dir / f"{participant_id}_mel.png"
        fig.tight_layout()
        fig.savefig(spec_path, dpi=100)
        plt.close(fig)
        created.append(spec_path)
    except ImportError:
        pass

    return created
