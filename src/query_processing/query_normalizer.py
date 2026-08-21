from __future__ import annotations

import logging
import re
from typing import Protocol

from src.common.settings import (
    ApplicationSettings,
    QueryNormalizationSettings,
    get_settings,
)
from src.query_processing.models import QueryNormalizationResult


LOGGER = logging.getLogger(__name__)


class TextCompletionProvider(Protocol):
    """Minimal provider capability required by query processing."""

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


class QueryNormalizer:
    """Conservatively normalize a query without domain vocabulary."""

    SYSTEM_PROMPT = """You normalize user queries for document retrieval.
Correct only clear spelling, typing, minor grammar, and word-order errors.
Use the whole query to infer obvious corrections and prefer a coherent common
phrase over interpreting a lowercase misspelling as a new proper noun.
Correct obvious misspellings of ordinary grammatical words as well. Treat an
unfamiliar token conservatively when casing, digits, or punctuation suggest
that it may be a name, acronym, identifier, version, or technical term.
Preserve the exact intent, language, proper nouns, acronyms, identifiers,
versions, product names, numbers, and unfamiliar technical terms.
Do not answer, explain, add facts, use markdown, or add quotation marks.
If uncertain or already correct, return the input unchanged.
Return only the normalized query.

Input: what is claud compoting
Output: what is cloud computing
Input: explan kubernetes archtectur
Output: explain Kubernetes architecture
Input: What is AKS?
Output: What is AKS?
Input: Explain XYZ-900 failover
Output: Explain XYZ-900 failover"""

    _ANSWER_PREFIX = re.compile(
        r"^(?:answer|explanation|the answer|here(?:'s| is)|certainly|sure)\s*[:,-]",
        re.IGNORECASE,
    )
    _LABEL_PREFIX = re.compile(
        r"^(?:normalized query|normalized|output)\s*:\s*",
        re.IGNORECASE,
    )

    def __init__(
        self,
        provider: TextCompletionProvider | None,
        settings: QueryNormalizationSettings,
    ) -> None:
        self._provider = provider
        self._settings = settings

    @property
    def provider(self) -> TextCompletionProvider | None:
        """Shared short-text provider for other query-processing components."""
        return self._provider

    @classmethod
    def from_settings(
        cls,
        settings: ApplicationSettings | None = None,
    ) -> "QueryNormalizer":
        application_settings = settings or get_settings()
        normalization = application_settings.query_processing.normalization
        provider: TextCompletionProvider | None = None

        if normalization.enabled:
            try:
                from src.generation.provider_factory import LLMProviderFactory

                candidate = LLMProviderFactory.create(
                    provider_name=normalization.provider,
                    model_name=normalization.model_name,
                    reasoning_effort=normalization.reasoning_effort,
                )
                if not hasattr(candidate, "complete_text"):
                    raise TypeError(
                        "The configured provider does not support text completion."
                    )
                provider = candidate  # type: ignore[assignment]
            except Exception as error:
                LOGGER.warning(
                    "Query normalizer provider unavailable; using original queries: %s",
                    error,
                )

        return cls(provider=provider, settings=normalization)

    def normalize(self, query: str) -> QueryNormalizationResult:
        original = str(query or "").strip()
        fallback = self._result(original, original)

        if not original or not self._settings.enabled or self._provider is None:
            return fallback

        try:
            output = self._provider.complete_text(
                system_prompt=self.SYSTEM_PROMPT,
                user_input=original,
                max_output_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
            )
            normalized = self._validate_output(original, output)
            if normalized is None:
                return fallback
            LOGGER.debug("Original query: %s", original)
            LOGGER.debug("Normalized query: %s", normalized)
            return self._result(original, normalized)
        except Exception as error:
            LOGGER.warning(
                "Query normalization failed; using original query: %s",
                error,
            )
            return fallback

    def _result(self, original: str, normalized: str) -> QueryNormalizationResult:
        provider = self._provider
        return QueryNormalizationResult(
            original_query=original,
            normalized_query=normalized,
            changed=normalized != original,
            provider=provider.provider_name if provider is not None else None,
            model_name=provider.model_name if provider is not None else None,
        )

    @classmethod
    def _validate_output(cls, original: str, output: str) -> str | None:
        candidate = str(output or "").strip()
        if not candidate:
            return None

        candidate = re.sub(r"^```(?:text)?\s*|\s*```$", "", candidate).strip()
        candidate = cls._LABEL_PREFIX.sub("", candidate).strip()
        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in "\"'"
        ):
            candidate = candidate[1:-1].strip()

        if not candidate or "\n" in candidate or "\r" in candidate:
            return None
        if len(candidate) > max(120, len(original) * 3):
            return None
        if cls._ANSWER_PREFIX.match(candidate):
            return None
        if candidate.startswith(("- ", "* ", "#")):
            return None
        return candidate
