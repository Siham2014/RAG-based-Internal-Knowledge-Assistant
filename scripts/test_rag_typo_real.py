from __future__ import annotations

from src.pipeline import RAGPipeline


QUESTION = "wat is claud compotin"


def main() -> None:
    rag = RAGPipeline.from_settings(
        language="en"
    )

    try:
        response = rag.answer(
            QUESTION
        )

        print("=" * 100)
        print("TYPO RAG TEST")
        print("=" * 100)

        print("Original question:")
        print(QUESTION)

        print()
        print("Normalization:")
        print(
            rag.last_query_normalization
        )

        print()
        print("Accepted:")
        print(
            response.accepted
        )

        print()
        print("Answer:")
        print(
            response.answer
        )

        print()
        print("Confidence:")
        print(
            response.confidence.confidence_score
        )

        print()
        print("Refusal reason:")
        print(
            response.refusal_reason
        )

        print()
        print("Sources:")

        for source in response.sources:
            print(
                "-",
                source.source,
                "|",
                source.reranker_score,
            )

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()