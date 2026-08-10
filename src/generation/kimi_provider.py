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


# ============================================================
# Valeurs de secours
# ============================================================

DEFAULT_KIMI_BASE_URL = "https://api.tokenrouter.com/v1"
DEFAULT_KIMI_MODEL = "moonshotai/kimi-k3-free"


class KimiProvider(BaseLLMProvider):
    """
    Fournisseur LLM compatible avec les modèles Kimi accessibles
    à travers une API au format OpenAI.

    Le provider peut fonctionner avec différents routeurs :

    - TokenRouter ;
    - Moonshot AI ;
    - tout autre endpoint compatible OpenAI.

    La configuration est chargée dans l'ordre suivant :

    1. arguments du constructeur ;
    2. variables du fichier .env ;
    3. valeurs par défaut.

    Variables prises en charge :

        TOKENROUTER_API_KEY
        MOONSHOT_API_KEY
        KIMI_BASE_URL
        KIMI_MODEL

    Aucune clé ni aucun modèle n'est codé directement dans
    le pipeline RAG.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        reasoning_effort: str | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        # override=True garantit que les valeurs actuelles
        # du fichier .env sont utilisées pendant les tests.
        load_dotenv(override=True)

        # ----------------------------------------------------
        # Modèle
        # ----------------------------------------------------

        selected_model_name = (
            model_name
            if model_name is not None
            else os.getenv(
                "KIMI_MODEL",
                DEFAULT_KIMI_MODEL,
            )
        )

        normalized_model_name = str(
            selected_model_name or ""
        ).strip()

        if not normalized_model_name:
            raise ValueError(
                "Aucun modèle Kimi n'est configuré. "
                "Ajoute KIMI_MODEL dans le fichier .env "
                "ou fournis model_name au constructeur."
            )

        # ----------------------------------------------------
        # Clé API
        # ----------------------------------------------------

        if api_key is not None:
            selected_api_key = str(
                api_key
            ).strip()

        else:
            selected_api_key = str(
                os.getenv(
                    "TOKENROUTER_API_KEY",
                    "",
                )
                or os.getenv(
                    "MOONSHOT_API_KEY",
                    "",
                )
            ).strip()

        if not selected_api_key:
            raise ValueError(
                "Aucune clé API Kimi n'est configurée. "
                "Ajoute TOKENROUTER_API_KEY ou "
                "MOONSHOT_API_KEY dans le fichier .env."
            )

        # ----------------------------------------------------
        # URL de base
        # ----------------------------------------------------

        selected_base_url = (
            base_url
            if base_url is not None
            else os.getenv(
                "KIMI_BASE_URL",
                DEFAULT_KIMI_BASE_URL,
            )
        )

        normalized_base_url = str(
            selected_base_url or ""
        ).strip().rstrip("/")

        if not normalized_base_url:
            raise ValueError(
                "KIMI_BASE_URL ne peut pas être vide."
            )

        if not normalized_base_url.startswith(
            (
                "https://",
                "http://",
            )
        ):
            raise ValueError(
                "KIMI_BASE_URL doit commencer par "
                "'https://' ou 'http://'."
            )

        # Refuser HTTP pour éviter d'envoyer une clé en clair.
        if normalized_base_url.startswith(
            "http://"
        ):
            raise ValueError(
                "KIMI_BASE_URL doit utiliser HTTPS pour "
                "protéger la clé API."
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Reasoning effort facultatif
        # ----------------------------------------------------

        normalized_reasoning_effort = (
            str(reasoning_effort)
            .strip()
            .lower()
            if reasoning_effort is not None
            else None
        )

        if normalized_reasoning_effort not in {
            None,
            "low",
            "medium",
            "high",
            "max",
        }:
            raise ValueError(
                "reasoning_effort doit être 'low', "
                "'medium', 'high', 'max' ou None."
            )

        # ----------------------------------------------------
        # Prompt builder
        # ----------------------------------------------------

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

        self._model_name = normalized_model_name
        self._base_url = normalized_base_url
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
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
        )

    # ========================================================
    # Propriétés
    # ========================================================

    @property
    def provider_name(self) -> str:
        return "kimi"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def reasoning_effort(
        self,
    ) -> str | None:
        return self._reasoning_effort

    # ========================================================
    # Extraction de la réponse et des métriques
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
    def _extract_answer(
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
                "L'API Kimi n'a retourné aucun choix."
            )

        first_choice = choices[0]

        if isinstance(
            first_choice,
            dict,
        ):
            message = first_choice.get(
                "message"
            )

        else:
            message = getattr(
                first_choice,
                "message",
                None,
            )

        if message is None:
            raise RuntimeError(
                "La réponse Kimi ne contient aucun message."
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

        # Certains endpoints peuvent retourner une liste
        # de blocs de contenu au lieu d'une simple chaîne.
        if isinstance(
            content,
            list,
        ):
            text_parts: list[str] = []

            for part in content:
                if isinstance(
                    part,
                    dict,
                ):
                    text = part.get(
                        "text"
                    )

                else:
                    text = getattr(
                        part,
                        "text",
                        None,
                    )

                if text:
                    text_parts.append(
                        str(text)
                    )

            normalized_content = "\n".join(
                text_parts
            ).strip()

        else:
            normalized_content = str(
                content or ""
            ).strip()

        if not normalized_content:
            raise RuntimeError(
                "Kimi a retourné une réponse vide."
            )

        return normalized_content

    # ========================================================
    # Gestion des erreurs
    # ========================================================

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

        parts = [
            "Échec de l'appel Kimi"
            + (
                f" avec le statut {status_code}"
                if status_code is not None
                else ""
            )
            + "."
        ]

        if detail:
            parts.append(
                f"Détail : {detail}"
            )

        if request_id:
            parts.append(
                f"Request ID : {request_id}"
            )

        return " ".join(
            parts
        )

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

        request_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": (
                request.max_output_tokens
            ),
            "stream": False,
        }

        # Certains endpoints refusent temperature=0.0.
        # Dans ce cas, le paramètre n'est pas envoyé.
        if request.temperature > 0.0:
            request_kwargs[
                "temperature"
            ] = request.temperature

        # Ce paramètre reste facultatif pour conserver
        # la compatibilité avec TokenRouter.
        if self._reasoning_effort is not None:
            request_kwargs[
                "reasoning_effort"
            ] = self._reasoning_effort

        start = time.perf_counter()

        try:
            response = (
                self._client
                .chat.completions
                .create(
                    **request_kwargs
                )
            )

        except AuthenticationError as error:
            raise RuntimeError(
                "Authentification Kimi refusée. "
                "Vérifie la clé API et KIMI_BASE_URL."
            ) from error

        except RateLimitError as error:
            raise RuntimeError(
                "Quota ou limite du fournisseur Kimi atteint."
            ) from error

        except APITimeoutError as error:
            raise RuntimeError(
                "L'appel Kimi a dépassé "
                f"{self._timeout_seconds} secondes."
            ) from error

        except APIConnectionError as error:
            raise RuntimeError(
                "Impossible de joindre l'API Kimi à "
                f"l'adresse {self._base_url}."
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
                "Kimi : "
                f"{type(error).__name__}: {error}"
            ) from error

        generation_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        answer = self._extract_answer(
            response
        )

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
    # Vérification locale
    # ========================================================

    def health_check(self) -> bool:
        """
        Vérifie uniquement la configuration locale.

        Aucun appel distant n'est effectué.
        """

        return bool(
            self._client
            and self._model_name
            and self._base_url
        )