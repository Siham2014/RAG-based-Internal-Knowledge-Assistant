from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryIntent(str, Enum):
    GREETING = "greeting"
    FOLLOW_UP = "follow_up"
    KNOWLEDGE_QUERY = "knowledge_query"
    OTHER = "other"


@dataclass(frozen=True)
class QueryNormalizationResult:
    """Result of the optional query-normalization step."""

    original_query: str
    normalized_query: str
    changed: bool
    provider: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class LanguageDetectionResult:
    language: str
    confidence: float

