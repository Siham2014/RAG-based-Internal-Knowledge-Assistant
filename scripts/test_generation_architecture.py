from __future__ import annotations

from src.generation import (
    CitationValidator,
    GenerationContext,
    GenerationRequest,
    LLMProviderFactory,
    RAGPromptBuilder,
)


def main() -> None:
    print("=" * 90)
    print("TEST DE L'ARCHITECTURE DE GÉNÉRATION")
    print("=" * 90)

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
            "is a shared responsibility between the "
            "cloud provider and the organization."
        ),
        document_format="markdown",
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
        max_output_tokens=200,
        temperature=0.0,
    )

    prompt_builder = RAGPromptBuilder()

    messages = prompt_builder.build_messages(
        request
    )

    print()
    print("SYSTEM PROMPT")
    print("-" * 90)
    print(
        messages[0]["content"]
    )

    print()
    print("USER PROMPT")
    print("-" * 90)
    print(
        messages[1]["content"]
    )

    provider = LLMProviderFactory.create(
        provider_name="mock",
        model_name="mock-rag-generator",
    )

    response = provider.generate(
        request
    )

    print()
    print("RÉPONSE DU MOCK PROVIDER")
    print("-" * 90)
    print(
        response.answer
    )

    validator = CitationValidator(
        require_at_least_one_citation=True
    )

    validation = validator.validate_response(
        response=response,
        contexts=request.contexts,
    )

    print()
    print("VALIDATION DES CITATIONS")
    print("-" * 90)

    for key, value in (
        validation
        .to_dict()
        .items()
    ):
        print(
            f"{key} : {value}"
        )

    if not validation.valid:
        raise RuntimeError(
            "Le test de validation des citations a échoué."
        )

    print()
    print("=" * 90)
    print("TEST TERMINÉ AVEC SUCCÈS")
    print("=" * 90)


if __name__ == "__main__":
    main()