"""Configurable tokenization."""

from __future__ import annotations

import re
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


class TextTokenizer:
    def __init__(self, method: str = "whitespace", model_name: str | None = None) -> None:
        self.method = method
        self.model_name = model_name
        self._hf = None

    def _load_hf(self) -> Any:
        if self._hf is not None:
            return self._hf
        try:
            from transformers import AutoTokenizer

            name = self.model_name or "bert-base-multilingual-cased"
            self._hf = AutoTokenizer.from_pretrained(name)
        except Exception:
            self._hf = None
        return self._hf

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        if self.method == "whitespace":
            return WHITESPACE_RE.split(text.strip())
        tok = self._load_hf()
        if tok is not None:
            return tok.tokenize(text)
        return WHITESPACE_RE.split(text.strip())

    def encode(self, text: str, max_length: int = 512) -> dict[str, Any]:
        tok = self._load_hf()
        if tok is not None:
            return tok(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        tokens = self.tokenize(text)[:max_length]
        return {"tokens": tokens, "length": len(tokens)}
