from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from src.conversation.memory import ConversationMemory
from src.conversation.models import ConversationMessage, MessageRole
from src.conversation.query_rewriter import ConversationQueryRewriter
from src.query_processing import IntentRouter, LanguageDetector, QueryIntent


if TYPE_CHECKING:
    from src.pipeline.rag_pipeline import RAGResponse, RAGPipeline


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationalResult:
    answer: str
    conversation_id: str
    intent: QueryIntent
    detected_language: str
    reply_language: str
    original_query: str
    normalized_query: str
    rewritten_query: str | None
    rag_response: "RAGResponse | None" = None


class ConversationalRAGService:
    """Fail-safe conversational layer around the unchanged retrieval stack."""

    def __init__(
        self,
        rag_pipeline: "RAGPipeline",
        memory: ConversationMemory,
        query_rewriter: ConversationQueryRewriter,
        language_detector: LanguageDetector | None = None,
        intent_router: IntentRouter | None = None,
    ) -> None:
        self._rag = rag_pipeline
        self._memory = memory
        self._rewriter = query_rewriter
        self._language_detector = language_detector or LanguageDetector()
        self._intent_router = intent_router or IntentRouter()

    def process(
        self,
        question: str,
        conversation_id: str | None = None,
        language: str = "auto",
        reply_language: str | None = None,
        response_style: str = "concise",
    ) -> ConversationalResult:
        original = str(question or "").strip()
        selected_id = str(conversation_id or uuid4()).strip()
        history = self._memory.get_history(selected_id)

        try:
            detected = self._language_detector.detect(original).language
        except Exception:
            detected = "en"

        requested = str(reply_language or language or "auto").strip().lower()
        effective_reply_language = detected if requested == "auto" else requested
        if effective_reply_language not in {"en", "fr", "ar"}:
            effective_reply_language = detected if detected in {"en", "fr", "ar"} else "en"

        try:
            intent = self._intent_router.route(original, bool(history))
        except Exception:
            intent = QueryIntent.KNOWLEDGE_QUERY

        if intent is QueryIntent.GREETING:
            answer = self._intent_router.greeting_response(effective_reply_language)
            self._remember(selected_id, original, answer)
            return ConversationalResult(
                answer=answer,
                conversation_id=selected_id,
                intent=intent,
                detected_language=detected,
                reply_language=effective_reply_language,
                original_query=original,
                normalized_query=original,
                rewritten_query=None,
            )

        needs_rewrite = intent is QueryIntent.FOLLOW_UP or detected != "en"
        rewritten_query: str | None = None
        retrieval_question = original
        normalize_query = True

        if needs_rewrite:
            rewrite = self._rewriter.rewrite(original, history)
            retrieval_question = rewrite.rewritten_query
            rewritten_query = retrieval_question
            normalize_query = False
            if not rewrite.changed and intent is QueryIntent.FOLLOW_UP:
                last_user = next(
                    (item.content for item in reversed(history) if item.role is MessageRole.USER),
                    "",
                )
                if last_user:
                    retrieval_question = f"Regarding '{last_user}': {original}"
                    rewritten_query = retrieval_question

        rag_response = self._rag.answer(
            retrieval_question,
            reply_language=effective_reply_language,
            response_style=response_style,
            normalize_query=normalize_query,
        )
        normalization = self._rag.last_query_normalization
        normalized_query = (
            normalization.normalized_query
            if normalization is not None
            else retrieval_question
        )
        self._remember(selected_id, original, rag_response.answer)
        return ConversationalResult(
            answer=rag_response.answer,
            conversation_id=selected_id,
            intent=intent,
            detected_language=detected,
            reply_language=effective_reply_language,
            original_query=original,
            normalized_query=normalized_query,
            rewritten_query=rewritten_query,
            rag_response=rag_response,
        )

    def _remember(self, conversation_id: str, question: str, answer: str) -> None:
        try:
            self._memory.append_message(
                conversation_id,
                ConversationMessage.create(MessageRole.USER, question),
            )
            self._memory.append_message(
                conversation_id,
                ConversationMessage.create(MessageRole.ASSISTANT, answer),
            )
        except Exception as error:
            LOGGER.warning("Conversation memory write failed: %s", error)

