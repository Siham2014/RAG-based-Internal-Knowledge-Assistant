from __future__ import annotations

import os

from dotenv import load_dotenv

from src.generation import (
    CitationValidator,
    GenerationContext,
    GenerationRequest,
    LLMGenerationError,
    LLMManager,
    LLMProviderFactory,
)


DEFAULT_TEST_MODEL = "moonshotai/kimi-k3-free"
DEFAULT_TEST_BASE_URL = "https://api.tokenrouter.com/v1"


def load_test_configuration() -> tuple[
    str,
    str,
]:
    """
    Charge la configuration du test depuis .env.

    Aucune clé n'est affichée dans le terminal.
    """

    load_dotenv(
        override=True
    )

    model_name = str(
        os.getenv(
            "KIMI_MODEL",
            DEFAULT_TEST_MODEL,
        )
    ).strip()

    base_url = str(
        os.getenv(
            "KIMI_BASE_URL",
            DEFAULT_TEST_BASE_URL,
        )
    ).strip().rstrip("/")

    tokenrouter_key = str(
        os.getenv(
            "TOKENROUTER_API_KEY",
            "",
        )
    ).strip()

    moonshot_key = str(
        os.getenv(
            "MOONSHOT_API_KEY",
            "",
        )
    ).strip()

    if not model_name:
        raise RuntimeError(
            "KIMI_MODEL est absent ou vide."
        )

    if not base_url:
        raise RuntimeError(
            "KIMI_BASE_URL est absent ou vide."
        )

    if not (
        tokenrouter_key
        or moonshot_key
    ):
        raise RuntimeError(
            "Aucune clé Kimi n'est configurée. "
            "Ajoute TOKENROUTER_API_KEY ou "
            "MOONSHOT_API_KEY dans .env."
        )

    print(
        "Clé API chargée :",
        "oui",
    )
    print(
        "Type de clé :",
        (
            "TOKENROUTER_API_KEY"
            if tokenrouter_key
            else "MOONSHOT_API_KEY"
        ),
    )

    return (
        model_name,
        base_url,
    )


def build_request() -> GenerationRequest:
    """
    Construit une requête courte avec une citation autorisée.
    """

    context = GenerationContext(
        rank=1,
        source=(
            "well-architected__"
            "sustainability__overview.md"
        ),
        chunk_id=(
            "fixed__1024__"
            "well-architected__"
            "sustainability__overview.md__"
            "chunk_0"
        ),
        content=(
            "Sustainability for workloads in the cloud "
            "is a shared responsibility between the cloud "
            "provider and the organization."
        ),
        document_format="markdown",
        page_number=None,
        source_url=None,
        reranker_score=0.981162,
    )

    return GenerationRequest(
        question=(
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        ),
        contexts=(
            context,
        ),
        language="en",
        max_output_tokens=100,
        temperature=0.0,
    )


def main() -> None:
    print("=" * 100)
    print("TEST UNITAIRE DU PROVIDER KIMI")
    print("=" * 100)

    (
        model_name,
        base_url,
    ) = load_test_configuration()

    request = build_request()

    provider = LLMProviderFactory.create(
        provider_name="kimi",
        model_name=model_name,
        base_url=base_url,
        reasoning_effort=None,
        timeout_seconds=120.0,
    )

    manager = LLMManager(
        primary_provider=provider,
        fallback_providers=(),
        max_attempts_per_provider=1,
        retry_delay_seconds=0.0,
    )

    print()
    print(
        "Provider :",
        manager.provider_name,
    )
    print(
        "Model :",
        manager.model_name,
    )
    print(
        "Base URL :",
        provider.base_url,
    )
    print(
        "Health :",
        manager.health_check(),
    )

    print()
    print(
        "Question :",
        request.question,
    )
    print(
        "Citation autorisée :",
        request.contexts[0].citation_id,
    )

    try:
        response = manager.generate(
            request
        )

    except LLMGenerationError as error:
        print()
        print("=" * 100)
        print("ÉCHEC DU TEST KIMI")
        print("=" * 100)
        print(str(error))

        print()
        print("Tentatives :")

        for attempt in error.attempts:
            print(
                attempt.to_dict()
            )

        raise SystemExit(1) from error

    print()
    print("=" * 100)
    print("RÉPONSE KIMI")
    print("=" * 100)
    print(response.answer)

    print()
    print(
        "Provider retourné :",
        response.provider,
    )
    print(
        "Modèle retourné :",
        response.model_name,
    )
    print(
        "Temps de génération :",
        response.generation_time_ms,
        "ms",
    )

    print()
    print("UTILISATION DES TOKENS")
    print("-" * 100)
    print(
        "Input tokens :",
        response.usage.input_tokens,
    )
    print(
        "Output tokens :",
        response.usage.output_tokens,
    )
    print(
        "Total tokens :",
        response.usage.total_tokens,
    )
    print(
        "Response ID :",
        response.raw_response_id,
    )

    validator = CitationValidator(
        require_at_least_one_citation=True
    )

    validation = (
        validator.validate_response(
            response=response,
            contexts=request.contexts,
        )
    )

    print()
    print("=" * 100)
    print("VALIDATION DES CITATIONS")
    print("=" * 100)

    for key, value in (
        validation.to_dict().items()
    ):
        print(
            f"{key} : {value}"
        )

    print()
    print("=" * 100)
    print("TENTATIVES DU LLM MANAGER")
    print("=" * 100)

    for attempt in manager.last_attempts:
        print(
            attempt.to_dict()
        )

    if not validation.valid:
        raise RuntimeError(
            "Kimi a répondu, mais la réponse ne "
            "contient pas une citation autorisée valide."
        )

    print()
    print("=" * 100)
    print("TEST KIMI RÉUSSI")
    print("=" * 100)


if __name__ == "__main__":
    main()