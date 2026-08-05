from __future__ import annotations

import time

from src.generation.base import (
    BaseLLMProvider,
)
from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)


class MockLLMProvider(BaseLLMProvider):
    """
    Fournisseur local de test.

    Aucun modèle n'est chargé et aucun appel API n'est effectué.
    """

    def __init__(
        self,
        model_name: str = "mock-rag-generator",
    ) -> None:
        self._model_name = str(
            model_name
        ).strip()

        if not self._model_name:
            raise ValueError(
                "model_name ne peut pas être vide."
            )

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        start = time.perf_counter()

        first_context = request.contexts[0]

        excerpt = " ".join(
            first_context.content.split()
        )[:300]

        answer = (
            f"Réponse de test basée sur le premier contexte : "
            f"{excerpt} "
            f"{first_context.citation_id}"
        )

        generation_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return GenerationResponse(
            answer=answer,
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(
                first_context.citation_id,
            ),
            usage=GenerationUsage(),
            generation_time_ms=round(
                generation_time_ms,
                4,
            ),
            raw_response_id=None,
        )