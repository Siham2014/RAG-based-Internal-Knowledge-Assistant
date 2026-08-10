from __future__ import annotations

from src.pipeline import RAGPipeline


TEST_CASES = (
    (
        "OPENAI_ANSWERABLE",
        True,
        (
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        ),
    ),
    (
        "OPENAI_OUT_OF_CORPUS",
        False,
        (
            "How do I configure Cisco IOS "
            "BGP route reflection?"
        ),
    ),
)


def print_separator() -> None:
    print("=" * 100)


def print_response(
    test_id: str,
    expected_answerable: bool,
    response,
) -> None:
    print()
    print_separator()
    print(test_id)
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
    print("CITATIONS")
    print("-" * 100)

    if response.citations:
        for citation in response.citations:
            print(citation)
    else:
        print("Aucune citation")

    print()
    print("CITATION VALIDATION")
    print("-" * 100)

    if response.citation_validation is None:
        print("Non exécutée")
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
    print("LLM USAGE")
    print("-" * 100)

    if response.generation is None:
        print(
            "Aucun appel LLM."
        )
    else:
        usage = response.generation.usage

        print(
            "Input tokens :",
            usage.input_tokens,
        )

        print(
            "Output tokens :",
            usage.output_tokens,
        )

        print(
            "Total tokens :",
            usage.total_tokens,
        )

        print(
            "Generation time :",
            response
            .generation
            .generation_time_ms,
            "ms",
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

    # Important :
    # une seule tentative pour limiter les coûts.
    rag.llm_manager.max_attempts_per_provider = 1
    rag.llm_manager.retry_delay_seconds = 0.0

    print_separator()
    print(
        "TEST CIBLÉ DU PIPELINE RAG AVEC OPENAI"
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

    try:
        for (
            test_id,
            expected_answerable,
            question,
        ) in TEST_CASES:

            print()
            print(
                "Question :",
                question,
            )

            response = rag.answer(
                question
            )

            print_response(
                test_id=test_id,
                expected_answerable=(
                    expected_answerable
                ),
                response=response,
            )

            # Vérifications simples
            if expected_answerable:
                if not response.accepted:
                    raise RuntimeError(
                        "La question répondable a été refusée."
                    )

                if response.provider != "openai":
                    raise RuntimeError(
                        "Le provider attendu est openai."
                    )

                if (
                    response.citation_validation
                    is None
                    or not response
                    .citation_validation
                    .valid
                ):
                    raise RuntimeError(
                        "La citation OpenAI n'est pas valide."
                    )

            else:
                if response.accepted:
                    raise RuntimeError(
                        "La question hors corpus "
                        "a été acceptée à tort."
                    )

                if response.generation is not None:
                    raise RuntimeError(
                        "Un appel OpenAI a été effectué "
                        "pour une question refusée."
                    )

        print()
        print_separator()
        print(
            "TEST OPENAI DU PIPELINE RAG RÉUSSI"
        )
        print_separator()

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()