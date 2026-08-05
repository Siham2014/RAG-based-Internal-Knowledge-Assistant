from __future__ import annotations

import json
from pathlib import Path

from src.pipeline import RetrievalPipeline


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "temp"
    / "pipeline_test"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "retrieval_pipeline_q001.json"
)

QUESTION = (
    "Who shares responsibility for the "
    "sustainability of cloud workloads?"
)


def main() -> None:
    print("=" * 100)
    print("TEST DU RETRIEVAL PIPELINE FINAL")
    print("=" * 100)

    pipeline = RetrievalPipeline(
        embedding_model_name=(
            "intfloat/e5-base-v2"
        ),
        reranker_model_name=(
            "BAAI/bge-reranker-base"
        ),
        device="auto",
        candidate_k=50,
        hybrid_top_k=20,
        final_top_k=5,
        rrf_constant=60,
        embedding_batch_size=1,
        reranker_batch_size=4,
        max_length=512,
    )

    print("Question :", QUESTION)
    print(
        "Embedding chargé avant appel :",
        pipeline.embedding_model_loaded,
    )
    print(
        "Reranker chargé avant appel :",
        pipeline.reranker_model_loaded,
    )
    print(
        "Device :",
        pipeline.reranker.device,
    )
    print()

    response = pipeline.retrieve(
        question=QUESTION,
        final_top_k=5,
    )

    print(
        "Embedding chargé après appel :",
        pipeline.embedding_model_loaded,
    )
    print(
        "Reranker chargé après appel :",
        pipeline.reranker_model_loaded,
    )

    print()
    print("=" * 100)
    print("TEMPS D'EXÉCUTION")
    print("=" * 100)
    print(
        "Embedding :",
        response.timings.embedding_time_ms,
        "ms",
    )
    print(
        "Hybrid retrieval :",
        response.timings.hybrid_retrieval_time_ms,
        "ms",
    )
    print(
        "Reranking :",
        response.timings.reranking_time_ms,
        "ms",
    )
    print(
        "Temps total :",
        response.timings.total_time_ms,
        "ms",
    )

    print()
    print("=" * 100)
    print("TOP 5 FINAL")
    print("=" * 100)

    for result in response.results:
        print()
        print("-" * 100)
        print("Rang final :", result.rank)
        print(
            "Ancien rang Hybrid :",
            result.hybrid_rank,
        )
        print("Chunk ID :", result.chunk_id)
        print("Source :", result.source)
        print(
            "Format :",
            result.document_format,
        )
        print(
            "Page :",
            result.page_number,
        )
        print(
            "Score reranker :",
            round(
                result.reranker_score,
                6,
            ),
        )
        print(
            "Score RRF :",
            round(
                result.rrf_score,
                8,
            ),
        )
        print(
            "Similarité cosinus :",
            (
                round(
                    result.cosine_similarity,
                    6,
                )
                if result.cosine_similarity
                is not None
                else None
            ),
        )
        print(
            "Rang vectoriel :",
            result.vector_rank,
        )
        print(
            "Rang BM25 :",
            result.bm25_rank,
        )
        print(
            "Contenu :",
            result.content[:500],
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            response.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Résultat JSON sauvegardé :",
        OUTPUT_PATH,
    )

    pipeline.unload_models()

    print()
    print("=" * 100)
    print("TEST TERMINÉ")
    print("=" * 100)


if __name__ == "__main__":
    main()