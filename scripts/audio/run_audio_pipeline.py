#!/usr/bin/env python3
"""Run audio engineering pipeline on discovered datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.audio.loaders.datasets import DAICAudioDataset, DVLOGAudioDataset
from ml_pipeline.audio.pipeline import AudioPipeline
from ml_pipeline.audio.statistics.stats import write_audio_statistics
from ml_pipeline.audio.visualization.plots import generate_audio_plots
from ml_pipeline.audio.utils.io import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="MindGraph++ audio pipeline")
    parser.add_argument("--dataset", choices=["daic_woz", "dvlog", "both"], default="daic_woz")
    parser.add_argument("--limit", type=int, default=0, help="Limit files (0=all)")
    args = parser.parse_args()

    pipeline = AudioPipeline()
    all_records: list[dict] = []
    if args.dataset in ("daic_woz", "both"):
        all_records.extend(DAICAudioDataset().discover())
    if args.dataset in ("dvlog", "both"):
        all_records.extend(DVLOGAudioDataset().discover())
    if args.limit > 0:
        all_records = all_records[: args.limit]

    print(f"Processing {len(all_records)} audio files...")
    results = pipeline.process_batch(all_records)
    meta_path = pipeline.save_metadata_parquet(results)
    stats_path = write_audio_statistics(results)
    print(f"Metadata: {meta_path}")
    print(f"Statistics: {stats_path}")
    print(f"Passed validation: {sum(1 for r in results if r.validation_passed)}/{len(results)}")

    if results and results[0].processed_path:
        audio, sr = load_audio(results[0].source_path)
        plots = generate_audio_plots(audio, sr, results[0].participant_id)
        if plots:
            print(f"Sample plots: {plots[0]}")


if __name__ == "__main__":
    main()
