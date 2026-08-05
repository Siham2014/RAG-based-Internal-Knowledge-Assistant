from __future__ import annotations

from src.common.settings import (
    get_settings,
)


def main() -> None:
    settings = get_settings(
        force_reload=True
    )

    print("=" * 80)
    print("CONFIGURATION CHARGÉE")
    print("=" * 80)

    print(
        "Fichier :",
        settings.source_path,
    )

    print(
        "Projet :",
        settings.project.name,
    )

    print(
        "Entreprise :",
        settings.corpus.company,
    )

    print(
        "Chunks :",
        settings.corpus.chunks_path,
    )

    print(
        "Chunking :",
        settings.chunking.strategy,
        settings.chunking.chunk_size,
        settings.chunking.overlap,
    )

    print(
        "Embedding :",
        settings.embedding.model_name,
    )

    print(
        "Dimension :",
        settings.embedding.dimension,
    )

    print(
        "Retriever :",
        settings.retrieval.vector_backend,
        "+",
        settings.retrieval.lexical_backend,
    )

    print(
        "Candidate K :",
        settings.retrieval.candidate_k,
    )

    print(
        "Hybrid Top K :",
        settings.retrieval.hybrid_top_k,
    )

    print(
        "Final Top K :",
        settings.retrieval.final_top_k,
    )

    print(
        "Reranker :",
        settings.reranking.model_name,
    )

    print(
        "Confidence activé :",
        settings.confidence.enabled,
    )

    print(
        "Confidence threshold :",
        settings.confidence.threshold,
    )

    print(
        "Génération activée :",
        settings.generation.enabled,
    )

    print(
        "BM25 :",
        settings.storage.bm25_index_path,
    )

    print(
        "API :",
        f"{settings.api.host}:"
        f"{settings.api.port}",
    )


if __name__ == "__main__":
    main()