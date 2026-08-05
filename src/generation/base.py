from __future__ import annotations

from abc import ABC, abstractmethod

from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class BaseLLMProvider(ABC):
    """
    Interface commune à tous les fournisseurs de génération.

    Le pipeline RAG dépend uniquement de cette interface,
    et non d'OpenAI, d'Ollama ou d'un modèle particulier.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Nom technique du fournisseur.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Nom du modèle utilisé.
        """

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Génère une réponse à partir de la question
        et des contextes documentaires.
        """

    def health_check(self) -> bool:
        """
        Vérification optionnelle du fournisseur.
        """

        return True