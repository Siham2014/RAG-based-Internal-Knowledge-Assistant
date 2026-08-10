from __future__ import annotations

from src.generation import (
    CitationValidator,
    GenerationContext,
    GenerationRequest,
    LLMGenerationError,
    LLMManager,
    LLMProviderFactory,
)


def main() -> None:
    context = GenerationContext(
        rank=1,
        source=(
            "well-architected__"
            "sustainability__overview.md"
        ),
        chunk_id=(
            "fixed__1024__well-architected__"
            "sustainability__overview.md__chunk_0"
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

    request = GenerationRequest(
        question=(
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        ),
        contexts=(
            context,
        ),
        language="en",
        max_output_tokens=60,
        temperature=0.0,
    )

    provider = LLMProviderFactory.create(
        provider_name="openai",
    )

    manager = LLMManager(
        primary_provider=provider,
        fallback_providers=(),
        max_attempts_per_provider=1,
        retry_delay_seconds=0.0,
    )

    print("=" * 100)
    print("TEST UNITAIRE OPENAI")
    print("=" * 100)

    print(
        "Provider :",
        manager.provider_name,
    )

    print(
        "Model :",
        manager.model_name,
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
        print("ÉCHEC OPENAI")
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
    print("RÉPONSE OPENAI")
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
    print("TOKENS")
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
    print("TENTATIVES LLM")
    print("=" * 100)

    for attempt in manager.last_attempts:
        print(
            attempt.to_dict()
        )

    if not validation.valid:
        raise RuntimeError(
            "OpenAI a répondu, mais la citation "
            "n'est pas valide."
        )

    print()
    print("=" * 100)
    print("TEST OPENAI RÉUSSI")
    print("=" * 100)


if __name__ == "__main__":
    main()