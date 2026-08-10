from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from src.generation.base import BaseLLMProvider
from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from src.generation.prompt_builder import RAGPromptBuilder


DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_REASONING_EFFORT = "minimal"

# Budget minimum de completion.
# Avec reasoning_effort="minimal", GPT-5 Nano ne devrait
# pratiquement plus consommer de tokens de raisonnement.
MINIMUM_COMPLETION_TOKENS = 100


class OpenAIProvider(BaseLLMProvider):
    """
    Fournisseur OpenAI pour le pipeline RAG.

    Configuration possible via :

    - arguments du constructeur ;
    - variables du fichier .env ;
    - valeurs par défaut.

    Variables utilisées :

        OPENAI_API_KEY
        OPENAI_MODEL

    Le provider est conçu pour rester économique :
    reasoning_effort="minimal" par défaut.

    health_check() ne réalise aucun appel distant.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        load_dotenv(
            override=True
        )

        # ====================================================
        # Modèle
        # ====================================================

        selected_model = (
            model_name
            if model_name is not None
            else os.getenv(
                "OPENAI_MODEL",
                DEFAULT_OPENAI_MODEL,
            )
        )

        normalized_model = str(
            selected_model or ""
        ).strip()

        if not normalized_model:
            raise ValueError(
                "Aucun modèle OpenAI n'est configuré."
            )

        # ====================================================
        # Clé API
        # ====================================================

        selected_api_key = (
            str(api_key).strip()
            if api_key is not None
            else str(
                os.getenv(
                    "OPENAI_API_KEY",
                    "",
                )
            ).strip()
        )

        if not selected_api_key:
            raise ValueError(
                "OPENAI_API_KEY est absent. "
                "Ajoute la clé OpenAI dans le fichier .env."
            )

        # ====================================================
        # Timeout
        # ====================================================

        try:
            normalized_timeout = float(
                timeout_seconds
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "timeout_seconds doit être un nombre."
            ) from error

        if normalized_timeout <= 0:
            raise ValueError(
                "timeout_seconds doit être supérieur à zéro."
            )

        # ====================================================
        # Reasoning effort
        # ====================================================

        normalized_reasoning_effort = (
            str(reasoning_effort)
            .strip()
            .lower()
            if reasoning_effort is not None
            else None
        )

        allowed_reasoning_efforts = {
            None,
            "minimal",
            "low",
            "medium",
            "high",
        }

        if (
            normalized_reasoning_effort
            not in allowed_reasoning_efforts
        ):
            raise ValueError(
                "reasoning_effort doit être "
                "'minimal', 'low', 'medium', "
                "'high' ou None."
            )

        # ====================================================
        # Prompt builder
        # ====================================================

        if (
            prompt_builder is not None
            and not isinstance(
                prompt_builder,
                RAGPromptBuilder,
            )
        ):
            raise TypeError(
                "prompt_builder doit être une instance "
                "de RAGPromptBuilder."
            )

        self._model_name = normalized_model
        self._timeout_seconds = normalized_timeout
        self._reasoning_effort = (
            normalized_reasoning_effort
        )

        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else RAGPromptBuilder()
        )

        self._client = OpenAI(
            api_key=selected_api_key,
            timeout=self._timeout_seconds,
            max_retries=0,
        )

    # ========================================================
    # Propriétés
    # ========================================================

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def reasoning_effort(
        self,
    ) -> str | None:
        return self._reasoning_effort

    # ========================================================
    # Utilitaires
    # ========================================================

    @staticmethod
    def _read_usage_value(
        usage: Any,
        field_name: str,
    ) -> int | None:
        if usage is None:
            return None

        if isinstance(
            usage,
            dict,
        ):
            value = usage.get(
                field_name
            )

        else:
            value = getattr(
                usage,
                field_name,
                None,
            )

        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _read_reasoning_tokens(
        usage: Any,
    ) -> int | None:
        """
        Extrait les reasoning tokens si OpenAI
        les retourne dans completion_tokens_details.
        """

        if usage is None:
            return None

        details = getattr(
            usage,
            "completion_tokens_details",
            None,
        )

        if details is None and isinstance(
            usage,
            dict,
        ):
            details = usage.get(
                "completion_tokens_details"
            )

        if details is None:
            return None

        if isinstance(
            details,
            dict,
        ):
            value = details.get(
                "reasoning_tokens"
            )

        else:
            value = getattr(
                details,
                "reasoning_tokens",
                None,
            )

        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _extract_answer(
        cls,
        response: Any,
    ) -> str:
        choices = getattr(
            response,
            "choices",
            None,
        )

        if choices is None and isinstance(
            response,
            dict,
        ):
            choices = response.get(
                "choices"
            )

        if not choices:
            raise RuntimeError(
                "OpenAI n'a retourné aucun choix."
            )

        first_choice = choices[0]

        if isinstance(
            first_choice,
            dict,
        ):
            message = first_choice.get(
                "message"
            )

            finish_reason = first_choice.get(
                "finish_reason"
            )

        else:
            message = getattr(
                first_choice,
                "message",
                None,
            )

            finish_reason = getattr(
                first_choice,
                "finish_reason",
                None,
            )

        if message is None:
            raise RuntimeError(
                "La réponse OpenAI ne contient "
                "aucun message."
            )

        if isinstance(
            message,
            dict,
        ):
            content = message.get(
                "content"
            )

        else:
            content = getattr(
                message,
                "content",
                None,
            )

        normalized_content = str(
            content or ""
        ).strip()

        if normalized_content:
            return normalized_content

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None and isinstance(
            response,
            dict,
        ):
            usage = response.get(
                "usage"
            )

        completion_tokens = (
            cls._read_usage_value(
                usage,
                "completion_tokens",
            )
        )

        reasoning_tokens = (
            cls._read_reasoning_tokens(
                usage
            )
        )

        raise RuntimeError(
            "OpenAI a retourné une réponse vide. "
            f"finish_reason={finish_reason}, "
            f"completion_tokens={completion_tokens}, "
            f"reasoning_tokens={reasoning_tokens}."
        )

    @staticmethod
    def _extract_api_error(
        error: APIStatusError,
    ) -> str:
        status_code = getattr(
            error,
            "status_code",
            None,
        )

        response = getattr(
            error,
            "response",
            None,
        )

        detail = ""

        if response is not None:
            try:
                detail = str(
                    response.text
                ).strip()

            except Exception:
                detail = ""

        request_id = getattr(
            error,
            "request_id",
            None,
        )

        message = (
            "Échec de l'appel OpenAI"
            + (
                f" avec le statut {status_code}"
                if status_code is not None
                else ""
            )
            + "."
        )

        if detail:
            message += (
                f" Détail : {detail}"
            )

        if request_id:
            message += (
                f" Request ID : {request_id}"
            )

        return message

    # ========================================================
    # Génération
    # ========================================================

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        if not isinstance(
            request,
            GenerationRequest,
        ):
            raise TypeError(
                "request doit être une instance "
                "de GenerationRequest."
            )

        messages = (
            self._prompt_builder
            .build_messages(
                request
            )
        )

        # ====================================================
        # Budget de sortie
        # ====================================================

        effective_max_tokens = max(
            int(
                request.max_output_tokens
            ),
            MINIMUM_COMPLETION_TOKENS,
        )

        request_kwargs: dict[
            str,
            Any,
        ] = {
            "model": self._model_name,
            "messages": messages,
            "max_completion_tokens": (
                effective_max_tokens
            ),
            "stream": False,
        }

        # ====================================================
        # Reasoning minimal
        # ====================================================

        if self._reasoning_effort is not None:
            request_kwargs[
                "reasoning_effort"
            ] = self._reasoning_effort

        # On n'envoie pas temperature=0.0.
        # Cela évite certains problèmes de compatibilité
        # avec les modèles GPT récents.
        if request.temperature > 0.0:
            request_kwargs[
                "temperature"
            ] = request.temperature

        # ====================================================
        # Appel API
        # ====================================================

        start = time.perf_counter()

        try:
            response = (
                self._client
                .chat
                .completions
                .create(
                    **request_kwargs
                )
            )

        except AuthenticationError as error:
            raise RuntimeError(
                "Authentification OpenAI refusée. "
                "Vérifie OPENAI_API_KEY."
            ) from error

        except RateLimitError as error:
            raise RuntimeError(
                "Quota ou limite OpenAI atteint."
            ) from error

        except APITimeoutError as error:
            raise RuntimeError(
                "L'appel OpenAI a dépassé "
                f"{self._timeout_seconds} secondes."
            ) from error

        except APIConnectionError as error:
            raise RuntimeError(
                "Impossible de joindre l'API OpenAI."
            ) from error

        except BadRequestError as error:
            raise RuntimeError(
                self._extract_api_error(
                    error
                )
            ) from error

        except APIStatusError as error:
            raise RuntimeError(
                self._extract_api_error(
                    error
                )
            ) from error

        except Exception as error:
            raise RuntimeError(
                "Erreur inattendue pendant la génération "
                "OpenAI : "
                f"{type(error).__name__}: {error}"
            ) from error

        generation_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        # ====================================================
        # Réponse
        # ====================================================

        answer = self._extract_answer(
            response
        )

        # ====================================================
        # Usage
        # ====================================================

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage is None and isinstance(
            response,
            dict,
        ):
            usage = response.get(
                "usage"
            )

        prompt_tokens = (
            self._read_usage_value(
                usage,
                "prompt_tokens",
            )
        )

        completion_tokens = (
            self._read_usage_value(
                usage,
                "completion_tokens",
            )
        )

        total_tokens = (
            self._read_usage_value(
                usage,
                "total_tokens",
            )
        )

        response_id = getattr(
            response,
            "id",
            None,
        )

        if (
            response_id is None
            and isinstance(
                response,
                dict,
            )
        ):
            response_id = response.get(
                "id"
            )

        return GenerationResponse(
            answer=answer,
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(),
            usage=GenerationUsage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            generation_time_ms=round(
                generation_time_ms,
                4,
            ),
            raw_response_id=(
                str(response_id)
                if response_id
                else None
            ),
        )

    # ========================================================
    # Health check gratuit
    # ========================================================

    def health_check(self) -> bool:
        """
        Vérification locale uniquement.

        Aucun appel API n'est effectué.
        Aucun token n'est consommé.
        """

        return bool(
            self._client
            and self._model_name
        )