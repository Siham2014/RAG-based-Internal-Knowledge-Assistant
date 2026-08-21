from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationContext:
    """
    Passage documentaire transmis au modèle de génération.
    """

    rank: int
    chunk_id: str
    source: str
    content: str

    source_url: str | None = None
    document_format: str | None = None
    page_number: int | None = None
    reranker_score: float | None = None

    @property
    def citation_id(self) -> str:
        """
        Citation stricte attendue dans la réponse.
        """

        return (
            f"[{self.source}:{self.chunk_id}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_url": self.source_url,
            "document_format": (
                self.document_format
            ),
            "page_number": self.page_number,
            "reranker_score": (
                self.reranker_score
            ),
            "citation_id": self.citation_id,
            "content": self.content,
        }


@dataclass(frozen=True)
class GenerationRequest:
    """
    Requête normalisée envoyée à un fournisseur LLM.
    """

    question: str
    contexts: tuple[GenerationContext, ...]

    language: str = "fr"
    max_output_tokens: int = 250
    temperature: float = 0.0
    response_style: str = "concise"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if not self.contexts:
            raise ValueError(
                "Au moins un contexte est nécessaire."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens doit être "
                "supérieur à zéro."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                "temperature doit être comprise "
                "entre 0 et 2."
            )

        if self.response_style not in {"concise", "detailed", "expert"}:
            raise ValueError(
                "response_style must be 'concise', 'detailed', or 'expert'."
            )


@dataclass(frozen=True)
class GenerationUsage:
    """
    Informations d'utilisation retournées par le fournisseur.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class GenerationResponse:
    """
    Réponse uniforme, quel que soit le fournisseur LLM.
    """

    answer: str
    provider: str
    model_name: str

    citations: tuple[str, ...]
    usage: GenerationUsage

    generation_time_ms: float
    raw_response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "provider": self.provider,
            "model_name": self.model_name,
            "citations": list(self.citations),
            "usage": self.usage.to_dict(),
            "generation_time_ms": (
                self.generation_time_ms
            ),
            "raw_response_id": (
                self.raw_response_id
            ),
        }
