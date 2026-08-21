from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.generation.base import (
    BaseLLMProvider,
)
from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMProviderAttempt:
    """
    Représente une tentative d'appel vers un fournisseur LLM.
    """

    provider: str
    model_name: str
    attempt_number: int

    success: bool
    duration_ms: float

    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "attempt_number": self.attempt_number,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class LLMGenerationError(RuntimeError):
    """
    Erreur levée lorsque tous les fournisseurs échouent.
    """

    def __init__(
        self,
        message: str,
        attempts: tuple[LLMProviderAttempt, ...],
    ) -> None:
        super().__init__(message)

        self.attempts = attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "attempts": [
                attempt.to_dict()
                for attempt in self.attempts
            ],
        }


class LLMResponseValidationError(RuntimeError):
    """Provider returned a response that violates the RAG output contract."""


class LLMManager:
    """
    Gestionnaire central des fournisseurs LLM.

    Responsabilités :

    - appeler le fournisseur principal ;
    - effectuer plusieurs tentatives ;
    - utiliser éventuellement des fournisseurs de secours ;
    - conserver l'historique des tentatives ;
    - retourner une erreur uniforme si tous les appels échouent.

    Le RAGPipeline dépend uniquement de cette classe et ne gère
    pas directement les erreurs propres à Hugging Face, OpenAI
    ou Ollama.
    """

    def __init__(
        self,
        primary_provider: BaseLLMProvider,
        fallback_providers: tuple[
            BaseLLMProvider,
            ...,
        ] = (),
        max_attempts_per_provider: int = 1,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if not isinstance(
            primary_provider,
            BaseLLMProvider,
        ):
            raise TypeError(
                "primary_provider doit hériter de "
                "BaseLLMProvider."
            )

        for provider in fallback_providers:
            if not isinstance(
                provider,
                BaseLLMProvider,
            ):
                raise TypeError(
                    "Chaque fournisseur de secours doit "
                    "hériter de BaseLLMProvider."
                )

        if max_attempts_per_provider <= 0:
            raise ValueError(
                "max_attempts_per_provider doit être "
                "supérieur à zéro."
            )

        if retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds doit être "
                "positif ou nul."
            )

        self.primary_provider = (
            primary_provider
        )

        self.fallback_providers = tuple(
            fallback_providers
        )

        self.max_attempts_per_provider = int(
            max_attempts_per_provider
        )

        self.retry_delay_seconds = float(
            retry_delay_seconds
        )

        self._last_attempts: tuple[
            LLMProviderAttempt,
            ...,
        ] = ()

    @property
    def providers(
        self,
    ) -> tuple[BaseLLMProvider, ...]:
        """
        Retourne le fournisseur principal suivi
        des fournisseurs de secours.
        """

        return (
            self.primary_provider,
            *self.fallback_providers,
        )

    @property
    def provider_name(self) -> str:
        """
        Nom du fournisseur principal.
        """

        return (
            self.primary_provider
            .provider_name
        )

    @property
    def model_name(self) -> str:
        """
        Modèle du fournisseur principal.
        """

        return (
            self.primary_provider
            .model_name
        )

    @property
    def last_attempts(
        self,
    ) -> tuple[LLMProviderAttempt, ...]:
        """
        Historique du dernier appel generate().
        """

        return self._last_attempts

    def health_check(
        self,
    ) -> dict[str, bool]:
        """
        Vérifie localement les fournisseurs enregistrés.

        Cette méthode ne doit pas nécessairement effectuer
        d'appel API ; son comportement dépend du provider.
        """

        health: dict[str, bool] = {}

        for provider in self.providers:
            key = (
                f"{provider.provider_name}:"
                f"{provider.model_name}"
            )

            try:
                health[key] = bool(
                    provider.health_check()
                )

            except Exception:
                health[key] = False

        return health

    def generate(
        self,
        request: GenerationRequest,
        response_validator: Callable[[GenerationResponse], bool] | None = None,
    ) -> GenerationResponse:
        """
        Tente la génération avec chaque fournisseur.

        Ordre :

        1. fournisseur principal ;
        2. nouvelles tentatives éventuelles ;
        3. fournisseurs de secours éventuels.

        La première réponse valide est retournée.
        """

        if not isinstance(
            request,
            GenerationRequest,
        ):
            raise TypeError(
                "request doit être une instance de "
                "GenerationRequest."
            )

        attempts: list[
            LLMProviderAttempt
        ] = []

        for provider_index, provider in enumerate(
            self.providers
        ):
            LOGGER.info("Generation provider: %s", provider.provider_name)
            for attempt_number in range(
                1,
                self.max_attempts_per_provider + 1,
            ):
                attempt_start = (
                    time.perf_counter()
                )

                try:
                    response = provider.generate(
                        request
                    )

                    duration_ms = (
                        time.perf_counter()
                        - attempt_start
                    ) * 1000

                    if not isinstance(
                        response,
                        GenerationResponse,
                    ):
                        raise TypeError(
                            "Le fournisseur n'a pas retourné "
                            "un GenerationResponse."
                        )

                    if not response.answer.strip():
                        raise ValueError(
                            "Le fournisseur a retourné "
                            "une réponse vide."
                        )

                    if (
                        response_validator is not None
                        and not response_validator(response)
                    ):
                        raise LLMResponseValidationError(
                            "The provider response failed grounded-output validation."
                        )

                    attempts.append(
                        LLMProviderAttempt(
                            provider=(
                                provider.provider_name
                            ),
                            model_name=(
                                provider.model_name
                            ),
                            attempt_number=(
                                attempt_number
                            ),
                            success=True,
                            duration_ms=round(
                                duration_ms,
                                4,
                            ),
                        )
                    )

                    self._last_attempts = tuple(
                        attempts
                    )

                    return response

                except Exception as error:
                    LOGGER.warning(
                        "%s generation failed: %s",
                        provider.provider_name.capitalize(),
                        str(error),
                    )
                    duration_ms = (
                        time.perf_counter()
                        - attempt_start
                    ) * 1000

                    attempts.append(
                        LLMProviderAttempt(
                            provider=(
                                provider.provider_name
                            ),
                            model_name=(
                                provider.model_name
                            ),
                            attempt_number=(
                                attempt_number
                            ),
                            success=False,
                            duration_ms=round(
                                duration_ms,
                                4,
                            ),
                            error_type=(
                                type(error).__name__
                            ),
                            error_message=str(
                                error
                            ),
                        )
                    )

                    is_last_attempt = (
                        attempt_number
                        >= self.max_attempts_per_provider
                    )

                    if not is_last_attempt:
                        if isinstance(error, LLMResponseValidationError):
                            break
                        time.sleep(
                            self.retry_delay_seconds
                        )

            # Petite pause avant un éventuel provider de secours.
            has_next_provider = (
                provider_index
                < len(self.providers) - 1
            )

            if has_next_provider:
                LOGGER.info(
                    "Trying fallback provider: %s",
                    self.providers[provider_index + 1].provider_name,
                )
                if self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds)

        self._last_attempts = tuple(
            attempts
        )

        provider_descriptions = ", ".join(
            (
                f"{provider.provider_name}"
                f"/{provider.model_name}"
            )
            for provider in self.providers
        )

        raise LLMGenerationError(
            (
                "La génération a échoué avec tous les "
                "fournisseurs configurés : "
                f"{provider_descriptions}."
            ),
            attempts=self._last_attempts,
        )
