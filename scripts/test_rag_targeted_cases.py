from __future__ import annotations

from src.pipeline import RAGPipeline


TEST_CASES = (
    (
        "RAG004",
        True,
        (
            "How are sustainability, reliability, security, "
            "and performance related in Azure workloads?"
        ),
    ),
    (
        "RAG007",
        False,
        (
            "How do I configure a VMware ESXi "
            "high-availability cluster?"
        ),
    ),
)


def print_separator() -> None:
    print("=" * 100)


def print_response(
    question_id: str,
    expected_answerable: bool,
    response,
) -> None:

    print()
    print_separator()
    print(question_id)
    print_separator()

    print(
        "Expected answerable :",
        expected_answerable,
    )

    print(
        "Confidence Gate accepted :",
        response.confidence.accepted,
    )

    print(
        "Final response accepted :",
        response.accepted,
    )

    print(
        "Confidence score :",
        round(
            response.confidence.confidence_score,
            6,
        ),
    )

    print(
        "Failed rules :",
        response.confidence.failed_rules,
    )

    print(
        "Provider :",
        response.provider,
    )

    print(
        "Model :",
        response.model_name,
    )

    print()

    print("ANSWER")
    print("-" * 100)

    print(response.answer)

    print()

    print("CITATION VALIDATION")
    print("-" * 100)

    if response.citation_validation is None:

        print("Not executed")

    else:

        for key, value in (
            response
            .citation_validation
            .to_dict()
            .items()
        ):
            print(
                f"{key} : {value}"
            )

    print()

    print("SOURCES")
    print("-" * 100)

    for source in response.sources:

        print(
            f"{source.rank}. "
            f"{source.source}"
        )

        print(
            f"Chunk : {source.chunk_id}"
        )

        print(
            "Reranker :",
            round(
                source.reranker_score,
                6,
            ),
        )

        print()

    print("TIMINGS")
    print("-" * 100)

    print(response.timings)

    print()

    print(
        "Refusal reason :",
        response.refusal_reason,
    )


def main() -> None:

    rag = RAGPipeline.from_settings(
        language="en"
    )

    rag.llm_manager.max_attempts_per_provider = 1
    rag.llm_manager.retry_delay_seconds = 0.0

    print_separator()

    print(
        "TEST CIBLÉ DU PIPELINE RAG"
    )

    print_separator()

    print(
        "Provider :",
        rag.llm_manager.provider_name,
    )

    print(
        "Model :",
        rag.llm_manager.model_name,
    )

    print(
        "Generation contexts :",
        rag.generation_contexts,
    )

    print(
        "Max output tokens :",
        rag.max_output_tokens,
    )

    print()

    try:

        for (
            question_id,
            expected_answerable,
            question,
        ) in TEST_CASES:

            print(
                "Question :",
                question,
            )

            response = rag.answer(
                question
            )

            print_response(
                question_id=question_id,
                expected_answerable=expected_answerable,
                response=response,
            )

    finally:

        rag.unload_models()


if __name__ == "__main__":
    main()