from __future__ import annotations

import logging
from typing import Protocol

from src.conversation.models import ConversationMessage, QueryRewriteResult


LOGGER = logging.getLogger(__name__)


class ShortTextProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_input: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str: ...


class ConversationQueryRewriter:
    """Produce one clean, standalone English retrieval query."""

    SYSTEM_PROMPT = """Rewrite the current user message as one standalone English
document-retrieval query. Use the short conversation context only to resolve
references. Correct clear spelling and grammar. Preserve intent, proper nouns,
acronyms, identifiers, versions, numbers, and unfamiliar technical terms.
Do not answer, explain, add facts, cite sources, or use markdown.
Return only the standalone retrieval query."""

    def __init__(
        self,
        provider: ShortTextProvider | None,
        max_output_tokens: int = 128,
        max_context_messages: int = 4,
        max_context_chars: int = 4000,
    ) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens
        self._max_context_messages = max_context_messages
        self._max_context_chars = max_context_chars

    def rewrite(
        self,
        query: str,
        history: tuple[ConversationMessage, ...] = (),
    ) -> QueryRewriteResult:
        original = str(query or "").strip()
        fallback = QueryRewriteResult(original, original, False)
        if not original or self._provider is None:
            return fallback

        context = history[-self._max_context_messages :]
        context_text = "\n".join(
            f"{message.role.value}: {message.content}" for message in context
        )
        context_text = context_text[-self._max_context_chars :]
        user_input = f"CONTEXT\n{context_text or '(none)'}\n\nCURRENT MESSAGE\n{original}"

        try:
            output = self._provider.complete_text(
                system_prompt=self.SYSTEM_PROMPT,
                user_input=user_input,
                max_output_tokens=self._max_output_tokens,
                temperature=0.0,
            )
            rewritten = str(output or "").strip().strip('"').strip("'").strip()
            if not rewritten or "\n" in rewritten:
                return fallback
            if len(rewritten) > max(160, len(original) * 8):
                return fallback
            return QueryRewriteResult(
                original_query=original,
                rewritten_query=rewritten,
                changed=rewritten != original,
                provider=self._provider.provider_name,
                model_name=self._provider.model_name,
            )
        except Exception as error:
            LOGGER.warning("Query rewrite failed; using current message: %s", error)
            return fallback
