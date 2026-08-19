from __future__ import annotations

from src.pipeline import RAGPipeline


QUESTIONS = [
    "What is cloud computing?",
    (
        "Explain the main economic advantages of cloud computing "
        "compared with traditional enterprise datacenters."
    ),
]


def main() -> None:
    rag = RAGPipeline.from_settings(
        language="en"
    )

    try:
        for number, question in enumerate(
            QUESTIONS,
            start=1,
        ):
            print()
            print("=" * 100)
            print(f"QUESTION {number}")
            print("=" * 100)
            print(question)

            response = rag.answer(
                question
            )

            print()
            print("-" * 100)
            print("FINAL RESPONSE")
            print("-" * 100)

            print(
                "Accepted :",
                response.accepted,
            )

            print(
                "Answer :",
                response.answer,
            )

            print(
                "Provider :",
                response.provider,
            )

            print(
                "Model :",
                response.model_name,
            )

            print(
                "Refusal reason :",
                response.refusal_reason,
            )

            print()
            print("-" * 100)
            print("CONFIDENCE")
            print("-" * 100)

            print(
                "Gate accepted :",
                response.confidence.accepted,
            )

            print(
                "Confidence score :",
                response.confidence.confidence_score,
            )

            print(
                "Failed rules :",
                response.confidence.failed_rules,
            )

            print()
            print("-" * 100)
            print("SOURCES")
            print("-" * 100)

            for source in response.sources:
                print()
                print(
                    "Rank :",
                    source.rank,
                )

                print(
                    "Source :",
                    source.source,
                )

                print(
                    "Chunk :",
                    source.chunk_id,
                )

                print(
                    "Reranker score :",
                    source.reranker_score,
                )

                print(
                    "RRF score :",
                    source.rrf_score,
                )

                print()
                print("CONTENT:")
                print(
                    source.content[:1200]
                )

                print("-" * 100)

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()