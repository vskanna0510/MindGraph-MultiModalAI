"""Mental health keyword lexicons."""

from __future__ import annotations

from pathlib import Path

DEFAULT_LEXICONS: dict[str, list[str]] = {
    "depression": ["depressed", "hopeless", "worthless", "empty", "sad", "down"],
    "anxiety": ["anxious", "worried", "nervous", "panic", "fear"],
    "stress": ["stressed", "overwhelmed", "pressure", "tension"],
    "sleep": ["insomnia", "sleep", "tired", "fatigue", "exhausted"],
    "isolation": ["alone", "lonely", "isolated", "withdrawn"],
    "loneliness": ["lonely", "loneliness", "no friends"],
    "burnout": ["burnout", "burned out", "drained"],
    "fatigue": ["fatigue", "exhausted", "drained", "weary"],
    "suicidal_ideation": ["suicide", "kill myself", "end my life", "want to die"],
}


def load_lexicons(lexicon_dir: Path | None = None) -> dict[str, list[str]]:
    lexicons = {k: list(v) for k, v in DEFAULT_LEXICONS.items()}
    if lexicon_dir and lexicon_dir.exists():
        for path in lexicon_dir.glob("*.txt"):
            lexicons[path.stem] = [
                line.strip().lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
    return lexicons


def extract_keywords(text: str, lexicons: dict[str, list[str]]) -> dict[str, list[str]]:
    lower = text.lower()
    found: dict[str, list[str]] = {}
    for category, terms in lexicons.items():
        hits = [t for t in terms if t in lower]
        if hits:
            found[category] = hits
    return found
