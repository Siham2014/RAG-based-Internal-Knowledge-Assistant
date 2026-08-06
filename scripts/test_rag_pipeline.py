from __future__ import annotations

from src.pipeline import RAGPipeline


def test_question(
    rag: RAGPipeline,
    question: str,
) -> None:
    print("=" * 100)
    print(question)
    print("=" * 100)

    response = rag.answer(
        question
    )

    print()
    print("Accepted :", response.accepted)
    print("Provider :", response.provider)
    print("Model    :", response.model_name)

    print()
    print("Answer")
    print("-" * 100)
    print(response.answer)

    print()
    print("Confidence")
    print("-" * 100)
    print(
        "Score :",
        round(
            response.confidence.confidence_score,
            6,
        ),
    )
    print(
        "Failed rules :",
        response.confidence.failed_rules,
    )

    print()
    print("Sources")
    print("-" * 100)

    for source in response.sources:
        print(
            f"{source.rank}. "
            f"{source.source}"
        )
        print(source.chunk_id)
        print()

    print("Citations")
    print("-" * 100)

    if response.citations:
        for citation in response.citations:
            print(citation)
    else:
        print("Aucune citation")

    print()
    print("Timings")
    print("-" * 100)
    print(response.timings)
    print()


def main() -> None:
    # Un seul pipeline pour toutes les questions.
    rag = RAGPipeline.from_settings(
        language="en"
    )

    try:
        test_question(
            rag,
            (
                "Who shares responsibility for the "
                "sustainability of cloud workloads?"
            ),
        )

        test_question(
            rag,
            (
                "How do I configure Cisco IOS "
                "BGP route reflection?"
            ),
        )

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()