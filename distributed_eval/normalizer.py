from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizerConfig:
    lowercase: bool = False
    remove_special_chars: bool = False
    strip_whitespace: bool = True
    normalize_unicode: bool = True


class TextNormalizer:
    """Frozen text normalizer for both documents and queries."""

    def __init__(self, config: dict):
        self.cfg = NormalizerConfig(
            lowercase=bool(config.get("lowercase", False)),
            remove_special_chars=bool(config.get("remove_special_chars", False)),
            strip_whitespace=bool(config.get("strip_whitespace", True)),
            normalize_unicode=bool(config.get("normalize_unicode", True)),
        )

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if self.cfg.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        if self.cfg.strip_whitespace:
            text = " ".join(text.split())
        if self.cfg.lowercase:
            text = text.lower()
        if self.cfg.remove_special_chars:
            # Keep alnum from all scripts and spaces.
            text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())

        return text

