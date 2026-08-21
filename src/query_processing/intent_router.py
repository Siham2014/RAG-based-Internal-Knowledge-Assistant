from __future__ import annotations

import re

from src.query_processing.followup_detector import FollowUpDetector
from src.query_processing.models import QueryIntent


class IntentRouter:
    """Route greetings locally and delegate contextuality detection."""

    _GREETING = re.compile(
        r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|"
        r"bonjour|bonsoir|salut|coucou|"
        r"مرحبا|مرحباً|أهلا|اهلا|السلام\s+عليكم)[!,.؟\s]*$",
        re.IGNORECASE,
    )

    def __init__(self, followup_detector: FollowUpDetector | None = None) -> None:
        self._followup_detector = followup_detector or FollowUpDetector()

    def route(self, message: str, has_history: bool) -> QueryIntent:
        text = str(message or "").strip()
        if self._GREETING.fullmatch(text):
            return QueryIntent.GREETING
        if self._followup_detector.is_follow_up(text, has_history):
            return QueryIntent.FOLLOW_UP
        return QueryIntent.KNOWLEDGE_QUERY

    @staticmethod
    def greeting_response(language: str) -> str:
        responses = {
            "fr": "Bonjour ! Comment puis-je vous aider avec votre base de connaissances ?",
            "ar": "مرحباً! كيف يمكنني مساعدتك في قاعدة المعرفة؟",
            "en": "Hello! How can I help you with your internal knowledge base?",
        }
        return responses.get(language, responses["en"])
