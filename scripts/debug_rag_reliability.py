from __future__ import annotations

from src.pipeline import RAGPipeline


QUESTION = "How can I improve Azure reliability?"


def main() -> None:
    rag = RAGPipeline.from_settings(
        language="en"
    )

    try:
        print("=" * 100)
        print("DEBUG RAG — AZURE RELIABILITY")
        print("=" * 100)

        print()
        print("QUESTION")
        print("-" * 100)
        print(QUESTION)

        response = rag.answer(
            QUESTION
        )

        print()
        print("=" * 100)
        print("FINAL RESPONSE")
        print("=" * 100)

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
        print("=" * 100)
        print("CONFIDENCE")
        print("=" * 100)

        print(
            "Accepted by gate :",
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
        print("=" * 100)
        print("RAW LLM GENERATION")
        print("=" * 100)

        if response.generation is None:
            print(
                "Aucune génération LLM disponible."
            )

        else:
            print(
                "LLM answer :",
                response.generation.answer,
            )

            print(
                "Provider :",
                response.generation.provider,
            )

            print(
                "Model :",
                response.generation.model_name,
            )

            print(
                "Input tokens :",
                response.generation.usage.input_tokens,
            )

            print(
                "Output tokens :",
                response.generation.usage.output_tokens,
            )

        print()
        print("=" * 100)
        print("CITATION VALIDATION")
        print("=" * 100)

        if response.citation_validation is None:
            print(
                "Validation non exécutée."
            )

        else:
            result = (
                response
                .citation_validation
                .to_dict()
            )

            for key, value in result.items():
                print(
                    f"{key} : {value}"
                )

        print()
        print("=" * 100)
        print("SOURCES")
        print("=" * 100)

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
                "Reranker :",
                source.reranker_score,
            )

            print(
                "Citation :",
                source.citation_id,
            )

            print()
            print(
                source.content[:1000]
            )

            print("-" * 100)

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()