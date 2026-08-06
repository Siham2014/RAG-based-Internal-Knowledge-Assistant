from __future__ import annotations

from src.generation import (
    GenerationContext,
    GenerationRequest,
    LLMManager,
    LLMProviderFactory,
)


def main() -> None:
    context = GenerationContext(
        rank=1,
        source="test_source.md",
        chunk_id="test_chunk_001",
        content=(
            "Cloud sustainability is a shared "
            "responsibility between the provider "
            "and the organization."
        ),
    )

    request = GenerationRequest(
        question=(
            "Who shares responsibility for "
            "cloud sustainability?"
        ),
        contexts=(
            context,
        ),
        language="en",
        max_output_tokens=100,
        temperature=0.0,
    )

    provider = LLMProviderFactory.create(
        provider_name="mock",
        model_name="mock-rag-generator",
    )

    manager = LLMManager(
        primary_provider=provider,
        max_attempts_per_provider=2,
        retry_delay_seconds=0.1,
    )

    response = manager.generate(
        request
    )

    print("=" * 80)
    print("RÉPONSE")
    print("=" * 80)
    print(response.answer)

    print()
    print("Provider :", response.provider)
    print("Model :", response.model_name)

    print()
    print("Tentatives :")

    for attempt in manager.last_attempts:
        print(attempt.to_dict())

    print()
    print("Health :", manager.health_check())


if __name__ == "__main__":
    main()