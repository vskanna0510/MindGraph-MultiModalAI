"""Transcript parsing, cleaning, and quality scoring."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

from ml_pipeline.datasets.parsers.types import TranscriptData, Utterance

WHITESPACE_RE = re.compile(r"\s+")
INVALID_UNICODE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Clean transcript text without removing semantic content."""
    text = INVALID_UNICODE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = text.replace("''", "'").replace('""', '"')
    return text


def _parse_daic_csv(path: Path) -> list[Utterance]:
    """Parse DAIC tab/comma-separated transcript."""
    utterances: list[Utterance] = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    delimiter = "\t" if "\t" in raw.splitlines()[0] else ","
    reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
    for order, row in enumerate(reader):
        speaker = row.get("speaker") or row.get("Speaker") or "unknown"
        start = float(row.get("start_time") or row.get("Start_Time") or 0)
        stop = float(row.get("stop_time") or row.get("Stop_Time") or start)
        text = clean_text(row.get("value") or row.get("Text") or row.get("text") or "")
        utterances.append(
            Utterance(
                speaker=speaker,
                start_time=start,
                stop_time=stop,
                text=text,
                order=order,
                duration=round(stop - start, 3),
            )
        )
    return utterances


def parse_transcript(path: Path) -> TranscriptData:
    """Parse transcript file into structured utterances."""
    if path.suffix.lower() == ".csv":
        utterances = _parse_daic_csv(path)
    else:
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        utterances = [
            Utterance(speaker="unknown", start_time=0.0, stop_time=0.0, text=text, order=0)
        ]

    words: list[str] = []
    for utt in utterances:
        words.extend(utt.text.lower().split())

    sentences = [u for u in utterances if u.text.strip()]
    vocab = set(words)
    avg_len = sum(len(u.text.split()) for u in sentences) / max(len(sentences), 1)
    unknown_tokens = sum(1 for w in words if not w.isalnum() and w not in {"'", "-"})

    quality = min(1.0, (len(utterances) / 10) * 0.3 + (len(vocab) / 100) * 0.3 + 0.4)
    if not utterances:
        quality = 0.0

    return TranscriptData(
        utterances=utterances,
        word_count=len(words),
        sentence_count=len(sentences),
        avg_sentence_length=round(avg_len, 2),
        vocabulary_size=len(vocab),
        unknown_tokens=unknown_tokens,
        quality_score=round(quality, 4),
    )


def export_transcript(data: TranscriptData, output_dir: Path, participant_id: str) -> TranscriptData:
    """Export transcript to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{participant_id}_transcript.json"
    csv_path = output_dir / f"{participant_id}_transcript.csv"

    payload = [
        {
            "speaker": u.speaker,
            "start_time": u.start_time,
            "stop_time": u.stop_time,
            "duration": u.duration,
            "text": u.text,
            "order": u.order,
            "relative_time": u.start_time,
            "absolute_time": u.start_time,
        }
        for u in data.utterances
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["order", "speaker", "start_time", "stop_time", "duration", "text"],
        )
        writer.writeheader()
        for u in data.utterances:
            writer.writerow(
                {
                    "order": u.order,
                    "speaker": u.speaker,
                    "start_time": u.start_time,
                    "stop_time": u.stop_time,
                    "duration": u.duration,
                    "text": u.text,
                }
            )

    data.json_path = json_path
    data.csv_path = csv_path
    return data
