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


MODEL_NAME = "google/gemma-2-2b-it"


def verify_environment() -> None:
    """
    Vérifie la présence du token sans afficher sa valeur.
    """

    load_dotenv()

    token = str(
        os.getenv(
            "HF_TOKEN",
            "",
        )
    ).strip()

    if not token:
        raise RuntimeError(
            "HF_TOKEN est absent. "
            "Ajoute-le dans le fichier .env."
        )

    print("HF_TOKEN chargé : oui")
    print(
        "Longueur du token :",
        len(token),
    )


def build_request() -> GenerationRequest:
    """
    Construit une requête courte afin de limiter
    la consommation de tokens.
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
    print("=" * 90)
    print("TEST RÉEL DU HUGGING FACE PROVIDER")
    print("=" * 90)

    verify_environment()

    request = build_request()

    provider = (
        LLMProviderFactory.create(
            provider_name="huggingface",
            model_name=MODEL_NAME,
            provider="featherless-ai",
            timeout_seconds=120.0,
        )
    )

    manager = LLMManager(
        primary_provider=provider,
        fallback_providers=(),
        # Une seule tentative afin de limiter les appels.
        max_attempts_per_provider=1,
        retry_delay_seconds=0.0,
    )

    print()
    print("Provider :", manager.provider_name)
    print("Modèle :", manager.model_name)
    print("Health :", manager.health_check())

    print()
    print("Question :", request.question)
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
        print("=" * 90)
        print("ÉCHEC DE LA GÉNÉRATION")
        print("=" * 90)
        print(str(error))

        print()
        print("Tentatives :")

        for attempt in error.attempts:
            print(attempt.to_dict())

        raise SystemExit(1) from error

    print()
    print("=" * 90)
    print("RÉPONSE HUGGING FACE")
    print("=" * 90)
    print(response.answer)

    print()
    print("Provider retourné :", response.provider)
    print("Modèle retourné :", response.model_name)
    print(
        "Temps de génération :",
        response.generation_time_ms,
        "ms",
    )
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
    print("=" * 90)
    print("VALIDATION DES CITATIONS")
    print("=" * 90)

    for key, value in (
        validation.to_dict().items()
    ):
        print(f"{key} : {value}")

    print()
    print("=" * 90)
    print("TENTATIVES DU LLM MANAGER")
    print("=" * 90)

    for attempt in manager.last_attempts:
        print(attempt.to_dict())

    if not validation.valid:
        raise RuntimeError(
            "Le fournisseur a répondu, mais la citation "
            "est absente ou invalide."
        )

    print()
    print("=" * 90)
    print("TEST HUGGING FACE RÉUSSI")
    print("=" * 90)


if __name__ == "__main__":
    main()