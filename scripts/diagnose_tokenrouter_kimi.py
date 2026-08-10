from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)


DEFAULT_BASE_URL = (
    "https://api.tokenrouter.com/v1"
)

DEFAULT_MODEL = (
    "moonshotai/kimi-k3-free"
)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 512


def main() -> None:
    load_dotenv(
        override=True
    )

    api_key = str(
        os.getenv(
            "TOKENROUTER_API_KEY",
            "",
        )
    ).strip()

    base_url = str(
        os.getenv(
            "KIMI_BASE_URL",
            DEFAULT_BASE_URL,
        )
    ).strip().rstrip("/")

    model_name = str(
        os.getenv(
            "KIMI_MODEL",
            DEFAULT_MODEL,
        )
    ).strip()

    if not api_key:
        raise RuntimeError(
            "TOKENROUTER_API_KEY est absent."
        )

    if not base_url:
        raise RuntimeError(
            "KIMI_BASE_URL est vide."
        )

    if not model_name:
        raise RuntimeError(
            "KIMI_MODEL est vide."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=0,
    )

    print("=" * 100)
    print("DIAGNOSTIC TOKENROUTER / KIMI")
    print("=" * 100)

    print(
        "Base URL :",
        base_url,
    )

    print(
        "Model :",
        model_name,
    )

    print(
        "Max tokens :",
        DEFAULT_MAX_TOKENS,
    )

    print(
        "Timeout :",
        DEFAULT_TIMEOUT_SECONDS,
        "s",
    )

    start = time.perf_counter()

    try:
        response = (
            client
            .chat
            .completions
            .create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer with exactly one "
                            "short sentence: "
                            "The test works."
                        ),
                    }
                ],
                max_tokens=(
                    DEFAULT_MAX_TOKENS
                ),
                stream=False,
            )
        )

    except APITimeoutError as error:
        elapsed = (
            time.perf_counter()
            - start
        )

        print()
        print(
            f"TIMEOUT après "
            f"{elapsed:.2f} s"
        )

        raise SystemExit(
            1
        ) from error

    except APIConnectionError as error:
        print()
        print(
            "Erreur de connexion :",
            error,
        )

        raise SystemExit(
            1
        ) from error

    except APIStatusError as error:
        print()
        print(
            "Erreur API :",
            error.status_code,
            error,
        )

        response_error = getattr(
            error,
            "response",
            None,
        )

        if response_error is not None:
            try:
                print(
                    "Détail :",
                    response_error.text,
                )
            except Exception:
                pass

        raise SystemExit(
            1
        ) from error

    except Exception as error:
        print()
        print(
            "Erreur inattendue :",
            type(error).__name__,
            error,
        )

        raise SystemExit(
            1
        ) from error

    elapsed = (
        time.perf_counter()
        - start
    )

    choices = getattr(
        response,
        "choices",
        None,
    )

    if not choices:
        raise RuntimeError(
            "Aucun choix n'a été retourné."
        )

    message = (
        choices[0]
        .message
    )

    visible_content = str(
        getattr(
            message,
            "content",
            "",
        )
        or ""
    ).strip()

    reasoning_content = str(
        getattr(
            message,
            "reasoning_content",
            "",
        )
        or ""
    ).strip()

    print()
    print("=" * 100)
    print("RÉPONSE VISIBLE")
    print("=" * 100)

    print(
        repr(
            visible_content
        )
    )

    print()
    print("=" * 100)
    print("RAISONNEMENT RETOURNÉ")
    print("=" * 100)

    if reasoning_content:
        print(
            reasoning_content
        )
    else:
        print(
            "Aucun reasoning_content."
        )

    print()
    print("=" * 100)
    print("MESSAGE COMPLET")
    print("=" * 100)

    try:
        print(
            message.model_dump()
        )
    except Exception:
        print(
            message
        )

    print()
    print("=" * 100)
    print("USAGE")
    print("=" * 100)

    print(
        response.usage
    )

    reasoning_tokens = None
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None

    if response.usage is not None:
        prompt_tokens = getattr(
            response.usage,
            "prompt_tokens",
            None,
        )

        completion_tokens = getattr(
            response.usage,
            "completion_tokens",
            None,
        )

        total_tokens = getattr(
            response.usage,
            "total_tokens",
            None,
        )

        details = getattr(
            response.usage,
            "completion_tokens_details",
            None,
        )

        if details is not None:
            reasoning_tokens = getattr(
                details,
                "reasoning_tokens",
                None,
            )

    print(
        "Prompt tokens :",
        prompt_tokens,
    )

    print(
        "Completion tokens :",
        completion_tokens,
    )

    print(
        "Reasoning tokens :",
        reasoning_tokens,
    )

    print(
        "Total tokens :",
        total_tokens,
    )

    print()
    print(
        f"Temps : {elapsed:.2f} s"
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC")
    print("=" * 100)

    if visible_content:
        print(
            "SUCCÈS : le modèle a produit "
            "une réponse visible."
        )

    elif reasoning_content:
        print(
            "Le modèle fonctionne, mais il a "
            "consommé son budget de sortie "
            "principalement en raisonnement."
        )

        print(
            "Il faudra augmenter le budget "
            "de tokens dans KimiProvider."
        )

    else:
        print(
            "Le modèle n'a produit ni texte visible "
            "ni raisonnement exploitable."
        )


if __name__ == "__main__":
    main()