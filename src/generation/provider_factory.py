from __future__ import annotations

from typing import Any

from src.generation.base import (
    BaseLLMProvider,
)
from src.generation.huggingface_provider import (
    HuggingFaceProvider,
)
from src.generation.mock_provider import (
    MockLLMProvider,
)
from src.generation.kimi_provider import (
    KimiProvider,
)
from src.generation.openai_provider import (
    OpenAIProvider,
)
from src.generation.qwen_provider import QwenProvider


class LLMProviderFactory:
    """
    Fabrique un fournisseur LLM à partir
    du nom défini dans settings.yaml.
    """

    _PROVIDERS: dict[
        str,
        type[BaseLLMProvider],
    ] = {
        "mock": MockLLMProvider,
        "huggingface": (
            HuggingFaceProvider
        ),
        "kimi": KimiProvider,
        "openai": OpenAIProvider,
        "qwen": QwenProvider,
    }

    @classmethod
    def register(
        cls,
        provider_name: str,
        provider_class: type[
            BaseLLMProvider
        ],
    ) -> None:
        normalized_name = str(
            provider_name
        ).strip().lower()

        if not normalized_name:
            raise ValueError(
                "provider_name ne peut pas "
                "être vide."
            )

        if not issubclass(
            provider_class,
            BaseLLMProvider,
        ):
            raise TypeError(
                "provider_class doit hériter "
                "de BaseLLMProvider."
            )

        cls._PROVIDERS[
            normalized_name
        ] = provider_class

    @classmethod
    def available_providers(
        cls,
    ) -> tuple[str, ...]:
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
        normalized_name = str(
            provider_name
        ).strip().lower()

        if not normalized_name:
            raise ValueError(
                "provider_name ne peut pas "
                "être vide."
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
                "Fournisseur LLM non "
                f"disponible : {provider_name}. "
                "Fournisseurs disponibles : "
                f"{available}."
            )

        return provider_class(
            **provider_kwargs
        )
