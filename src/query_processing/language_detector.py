from __future__ import annotations

import re

from src.query_processing.models import LanguageDetectionResult


class LanguageDetector:
    """Small offline detector optimized for English, French, and Arabic."""

    _ARABIC = re.compile(r"[\u0600-\u06ff]")
    _FRENCH_CHARS = re.compile(r"[àâçéèêëîïôùûüÿœæ]", re.IGNORECASE)
    _FRENCH_CONTRACTION = re.compile(
        r"\b(?:c|d|j|l|m|n|qu|s|t)['’][a-zà-ÿ]",
        re.IGNORECASE,
    )
    _FRENCH_MARKERS = frozenset(
        {
            "au", "aux", "avec", "ce", "ces", "cette", "cinq", "dans",
            "de", "des", "du", "elle", "en", "est", "et", "ils", "je",
            "la", "le", "les", "mais", "nous", "ou", "par", "pas", "plus",
            "pour", "que", "quel", "quelle", "quelles", "quels", "qui",
            "sans", "se", "ses", "sont", "sur", "tu", "un", "une", "vous",
            "quoi", "comment", "pourquoi", "bonjour", "bonsoir", "salut",
        }
    )

    def detect(self, text: str) -> LanguageDetectionResult:
        normalized = str(text or "").strip()
        if self._ARABIC.search(normalized):
            return LanguageDetectionResult("ar", 0.99)
        if (
            self._FRENCH_CHARS.search(normalized)
            or self._FRENCH_CONTRACTION.search(normalized)
        ):
            return LanguageDetectionResult("fr", 0.90)
        words = set(re.findall(r"[a-zA-ZÀ-ÿŒœÆæ]+", normalized.lower()))
        matches = len(words & self._FRENCH_MARKERS)
        if matches >= 2 or words & {"bonjour", "bonsoir", "salut"}:
            return LanguageDetectionResult("fr", 0.80)
        return LanguageDetectionResult("en", 0.60)
