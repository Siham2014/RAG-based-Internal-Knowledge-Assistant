from __future__ import annotations

from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.common.settings import get_settings


QUESTION = "How can I improve Azure reliability?"


def main() -> None:
    settings = get_settings()

    pipeline = RetrievalPipeline.from_settings(
        settings
    )

    print("=" * 100)
    print("DEBUG RETRIEVAL")
    print("=" * 100)
    print("Question :", QUESTION)
    print()

    response = pipeline.retrieve(
        question=QUESTION,
        final_top_k=5,
    )

    print("=" * 100)
    print("TOP RESULTS")
    print("=" * 100)

    for result in response.results:
        print()
        print("-" * 100)

        print(
            "Rank :",
            result.rank,
        )

        print(
            "Source :",
            result.source,
        )

        print(
            "Chunk :",
            result.chunk_id,
        )

        print(
            "RRF score :",
            result.rrf_score,
        )

        print(
            "Reranker score :",
            result.reranker_score,
        )

        print(
            "Page :",
            result.page_number,
        )

        print()
        print("CONTENT")
        print("-" * 100)

        content = str(
            result.content or ""
        )

        print(
            content[:1200]
        )

    pipeline.unload_models()


if __name__ == "__main__":
    main()