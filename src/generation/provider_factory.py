from __future__ import annotations

from typing import Any

from src.generation.base import (
    BaseLLMProvider,
)
from src.generation.mock_provider import (
    MockLLMProvider,
)


class LLMProviderFactory:
    """
    Fabrique le fournisseur LLM demandé.

    Les fournisseurs sont sélectionnés par leur nom,
    sans modifier le pipeline RAG.
    """

    _PROVIDERS: dict[
        str,
        type[BaseLLMProvider],
    ] = {
        "mock": MockLLMProvider,
    }

    @classmethod
    def register(
        cls,
        provider_name: str,
        provider_class: type[
            BaseLLMProvider
        ],
    ) -> None:
        """
        Enregistre dynamiquement un nouveau fournisseur.
        """

        normalized_name = str(
            provider_name
        ).strip().lower()

        if not normalized_name:
            raise ValueError(
                "provider_name ne peut pas être vide."
            )

        if not issubclass(
            provider_class,
            BaseLLMProvider,
        ):
            raise TypeError(
                "provider_class doit hériter de "
                "BaseLLMProvider."
            )

        cls._PROVIDERS[
            normalized_name
        ] = provider_class

    @classmethod
    def available_providers(
        cls,
    ) -> tuple[str, ...]:
        """
        Retourne les fournisseurs actuellement disponibles.
        """

        return tuple(
            sorted(
                cls._PROVIDERS.keys()
            )
        )

    @classmethod
    def create(
        cls,
        provider_name: str,
        **provider_kwargs: Any,
    ) -> BaseLLMProvider:
        """
        Crée un fournisseur à partir de son nom.
        """

        normalized_name = str(
            provider_name
        ).strip().lower()

        if not normalized_name:
            raise ValueError(
                "provider_name ne peut pas être vide."
            )

        provider_class = (
            cls._PROVIDERS.get(
                normalized_name
            )
        )

        if provider_class is None:
            available = ", ".join(
                cls.available_providers()
            )

            raise ValueError(
                "Fournisseur LLM non disponible : "
                f"{provider_name}. "
                f"Fournisseurs disponibles : {available}."
            )

        return provider_class(
            **provider_kwargs
        )