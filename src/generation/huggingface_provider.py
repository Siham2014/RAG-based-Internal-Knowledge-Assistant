from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import (
    HfHubHTTPError,
    InferenceTimeoutError,
)

from src.generation.base import (
    BaseLLMProvider,
)
from src.generation.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
)
from src.generation.prompt_builder import (
    RAGPromptBuilder,
)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Fournisseur de génération utilisant les
    Hugging Face Inference Providers.

    Le modèle n'est pas téléchargé localement.

    L'authentification utilise la variable
    d'environnement HF_TOKEN.

    Le fournisseur d'inférence peut être :

    - auto ;
    - featherless-ai ;
    - ou un autre fournisseur supporté par
      Hugging Face.
    """

    def __init__(
        self,
        model_name: str,
        api_token: str | None = None,
        provider: str | None = None,
        timeout_seconds: float = 120.0,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        load_dotenv()

        normalized_model_name = str(
            model_name or ""
        ).strip()

        if not normalized_model_name:
            raise ValueError(
                "model_name ne peut pas être vide."
            )

        selected_token = (
            str(api_token).strip()
            if api_token is not None
            else str(
                os.getenv(
                    "HF_TOKEN",
                    "",
                )
            ).strip()
        )

        if not selected_token:
            raise ValueError(
                "Le token Hugging Face est absent. "
                "Ajoute HF_TOKEN dans le fichier .env."
            )

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
                "timeout_seconds doit être supérieur "
                "à zéro."
            )

        normalized_provider = (
            str(provider).strip()
            if provider is not None
            else "auto"
        )

        if not normalized_provider:
            normalized_provider = "auto"

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

        self._model_name = (
            normalized_model_name
        )

        self._provider = (
            normalized_provider
        )

        self._timeout_seconds = (
            normalized_timeout
        )

        self._prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else RAGPromptBuilder()
        )

        self._client = InferenceClient(
            api_key=selected_token,
            provider=self._provider,
            timeout=self._timeout_seconds,
        )

    # ========================================================
    # Propriétés
    # ========================================================

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def inference_provider(self) -> str:
        return self._provider

    # ========================================================
    # Préparation des messages
    # ========================================================

    def _prepare_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        """
        Construit des messages compatibles avec le modèle.

        Certains modèles, notamment Gemma 2, ne prennent pas
        directement en charge le rôle system.

        Pour les modèles Gemma, les instructions système sont
        donc fusionnées avec le message utilisateur afin de
        produire un seul message de rôle user.
        """

        messages = (
            self._prompt_builder
            .build_messages(
                request
            )
        )

        normalized_model_name = (
            self._model_name
            .strip()
            .casefold()
        )

        is_gemma_model = (
            "gemma" in normalized_model_name
        )

        if not is_gemma_model:
            return messages

        system_parts: list[str] = []
        user_parts: list[str] = []
        assistant_parts: list[str] = []

        for message in messages:
            role = str(
                message.get(
                    "role",
                    "",
                )
            ).strip().lower()

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if not content:
                continue

            if role == "system":
                system_parts.append(
                    content
                )

            elif role == "user":
                user_parts.append(
                    content
                )

            elif role == "assistant":
                assistant_parts.append(
                    content
                )

        merged_parts: list[str] = []

        if system_parts:
            merged_parts.append(
                "SYSTEM INSTRUCTIONS\n"
                + "\n\n".join(
                    system_parts
                )
            )

        if user_parts:
            merged_parts.append(
                "USER REQUEST\n"
                + "\n\n".join(
                    user_parts
                )
            )

        if assistant_parts:
            merged_parts.append(
                "PREVIOUS ASSISTANT CONTENT\n"
                + "\n\n".join(
                    assistant_parts
                )
            )

        merged_content = "\n\n".join(
            merged_parts
        ).strip()

        if not merged_content:
            raise RuntimeError(
                "Aucun message valide n'a été construit "
                "pour le modèle Gemma."
            )

        return [
            {
                "role": "user",
                "content": merged_content,
            }
        ]

    # ========================================================
    # Utilitaires
    # ========================================================

    @staticmethod
    def _read_usage_value(
        usage: Any,
        field_name: str,
    ) -> int | None:
        """
        Lit une valeur dans un objet usage ou un dictionnaire.
        """

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
        """
        Extrait le texte de la première réponse retournée.
        """

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
                "Hugging Face n'a retourné "
                "aucun choix de réponse."
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
                "La réponse Hugging Face ne contient "
                "pas de message."
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

        if not normalized_content:
            raise RuntimeError(
                "Le modèle Hugging Face a retourné "
                "une réponse vide."
            )

        return normalized_content

    @staticmethod
    def _extract_error_details(
        error: HfHubHTTPError,
    ) -> tuple[
        int | None,
        str,
        str | None,
    ]:
        """
        Extrait le statut HTTP, le corps de réponse
        et l'identifiant de requête.
        """

        error_response = getattr(
            error,
            "response",
            None,
        )

        status_code = getattr(
            error_response,
            "status_code",
            None,
        )

        response_text = ""

        if error_response is not None:
            try:
                response_text = str(
                    error_response.text
                ).strip()

            except Exception:
                response_text = ""

        request_id = getattr(
            error,
            "request_id",
            None,
        )

        if request_id is None and (
            error_response is not None
        ):
            headers = getattr(
                error_response,
                "headers",
                {},
            )

            if headers:
                request_id = (
                    headers.get(
                        "x-request-id"
                    )
                    or headers.get(
                        "x-amzn-trace-id"
                    )
                )

        return (
            status_code,
            response_text,
            (
                str(request_id)
                if request_id
                else None
            ),
        )

    @classmethod
    def _build_http_error_message(
        cls,
        error: HfHubHTTPError,
    ) -> str:
        """
        Produit un message lisible à partir
        d'une erreur HTTP Hugging Face.
        """

        (
            status_code,
            response_text,
            request_id,
        ) = cls._extract_error_details(
            error
        )

        detail_parts: list[str] = []

        if response_text:
            detail_parts.append(
                f"Détail : {response_text}"
            )

        if request_id:
            detail_parts.append(
                f"Request ID : {request_id}"
            )

        detail_suffix = (
            " " + " | ".join(
                detail_parts
            )
            if detail_parts
            else ""
        )

        if status_code == 400:
            return (
                "La requête Hugging Face a été "
                "refusée par le fournisseur."
                + detail_suffix
            )

        if status_code == 401:
            return (
                "Authentification Hugging Face refusée. "
                "Vérifie HF_TOKEN."
                + detail_suffix
            )

        if status_code == 403:
            return (
                "Le token Hugging Face n'a pas "
                "l'autorisation d'utiliser ce modèle "
                "ou ce fournisseur d'inférence."
                + detail_suffix
            )

        if status_code == 404:
            return (
                "Le modèle ou l'endpoint Hugging Face "
                "est introuvable."
                + detail_suffix
            )

        if status_code == 422:
            return (
                "Les paramètres de génération ont été "
                "refusés par le fournisseur Hugging Face."
                + detail_suffix
            )

        if status_code == 429:
            return (
                "La limite ou le quota Hugging Face "
                "a été atteint."
                + detail_suffix
            )

        if status_code == 503:
            return (
                "Le modèle ou le fournisseur Hugging Face "
                "est temporairement indisponible."
                + detail_suffix
            )

        return (
            "Échec de l'appel Hugging Face"
            + (
                f" avec le statut {status_code}"
                if status_code is not None
                else ""
            )
            + "."
            + detail_suffix
        )

    # ========================================================
    # Génération
    # ========================================================

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Génère une réponse à partir d'une GenerationRequest.
        """

        if not isinstance(
            request,
            GenerationRequest,
        ):
            raise TypeError(
                "request doit être une instance "
                "de GenerationRequest."
            )

        messages = self._prepare_messages(
            request
        )

        generation_kwargs: dict[
            str,
            Any,
        ] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": (
                request.max_output_tokens
            ),
            "stream": False,
        }

        # Certains fournisseurs refusent temperature=0.0.
        # Dans ce cas, ce paramètre n'est pas envoyé.
        if request.temperature > 0.0:
            generation_kwargs[
                "temperature"
            ] = request.temperature

        start = time.perf_counter()

        try:
            response = (
                self._client
                .chat_completion(
                    **generation_kwargs
                )
            )

        except InferenceTimeoutError as error:
            raise RuntimeError(
                "Le fournisseur Hugging Face a dépassé "
                f"le délai de {self._timeout_seconds} secondes."
            ) from error

        except HfHubHTTPError as error:
            raise RuntimeError(
                self._build_http_error_message(
                    error
                )
            ) from error

        except Exception as error:
            raise RuntimeError(
                "Erreur inattendue pendant la génération "
                "Hugging Face : "
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

        raw_response_id = getattr(
            response,
            "id",
            None,
        )

        if (
            raw_response_id is None
            and isinstance(
                response,
                dict,
            )
        ):
            raw_response_id = response.get(
                "id"
            )

        return GenerationResponse(
            answer=answer,
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(),
            usage=GenerationUsage(
                input_tokens=prompt_tokens,
                output_tokens=(
                    completion_tokens
                ),
                total_tokens=total_tokens,
            ),
            generation_time_ms=round(
                generation_time_ms,
                4,
            ),
            raw_response_id=(
                str(raw_response_id)
                if raw_response_id
                else None
            ),
        )

    # ========================================================
    # Vérification locale
    # ========================================================

    def health_check(self) -> bool:
        """
        Vérifie seulement que le client, le modèle et le
        fournisseur sont configurés.

        Aucun appel distant n'est réalisé.
        """

        return bool(
            self._client
            and self._model_name
            and self._provider
        )