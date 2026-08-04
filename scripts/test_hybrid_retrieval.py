from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from src.retrieval.hybrid_rrf_retriever import (
    HybridRRFRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_30_questions.csv"
)

QUERY_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "e5_base_query_embeddings.npy"
)

CANDIDATE_K = 50
FINAL_TOP_K = 10
RRF_CONSTANT = 60


def load_first_question(
    path: Path,
) -> dict[str, str]:
    """
    Charge la première question du fichier CSV.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        row = next(
            reader,
            None,
        )

    if row is None:
        raise ValueError(
            "Le fichier d'évaluation est vide."
        )

    return {
        key: str(value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def canonical_filename(
    source: str,
) -> str:
    """
    Normalise un nom de fichier source.
    """

    return (
        str(source)
        .replace("\\", "/")
        .split("/")[-1]
        .lower()
    )


def main() -> None:
    """
    Teste la recherche hybride sur Q001.
    """

    question = load_first_question(
        EVALUATION_PATH
    )

    query_embeddings = np.load(
        QUERY_EMBEDDINGS_PATH,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    if query_embeddings.shape != (
        30,
        768,
    ):
        raise ValueError(
            "Forme incorrecte pour les embeddings : "
            f"{query_embeddings.shape}"
        )

    query_embedding = (
        query_embeddings[0]
    )

    retriever = HybridRRFRetriever(
        candidate_k=CANDIDATE_K,
        final_top_k=FINAL_TOP_K,
        rrf_constant=RRF_CONSTANT,
    )

    vector_count = (
        retriever.count_vector_chunks()
    )

    bm25_count = (
        retriever.count_bm25_chunks()
    )

    if vector_count != bm25_count:
        raise ValueError(
            "Les deux moteurs ne contiennent pas "
            "le même nombre de chunks."
        )

    print("=" * 100)
    print(
        "TEST DU RETRIEVAL HYBRIDE — PGVECTOR + BM25 + RRF"
    )
    print("=" * 100)

    print(
        "Chunks pgvector :",
        vector_count,
    )

    print(
        "Chunks BM25 :",
        bm25_count,
    )

    print(
        "Question ID :",
        question["id"],
    )

    print(
        "Question :",
        question["question"],
    )

    print(
        "Source attendue :",
        question["expected_source"],
    )

    print(
        "Candidats par moteur :",
        CANDIDATE_K,
    )

    print(
        "Top K final :",
        FINAL_TOP_K,
    )

    print(
        "Constante RRF :",
        RRF_CONSTANT,
    )

    start_time = time.perf_counter()

    results = retriever.search(
        question=question["question"],
        query_embedding=query_embedding,
        final_top_k=FINAL_TOP_K,
        candidate_k=CANDIDATE_K,
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    print()
    print(
        f"Recherche hybride terminée en "
        f"{elapsed_ms:.3f} ms"
    )

    print()
    print("=" * 100)
    print("TOP 10 HYBRIDE")
    print("=" * 100)

    for result in results:

        print()
        print("-" * 100)

        print(
            "Rang final :",
            result.rank,
        )

        print(
            "Chunk ID :",
            result.chunk_id,
        )

        print(
            "Source :",
            result.source,
        )

        print(
            "Format :",
            result.document_format,
        )

        print(
            "Page :",
            result.page_number,
        )

        print(
            "Rang pgvector :",
            result.vector_rank,
        )

        print(
            "Rang BM25 :",
            result.bm25_rank,
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
            "Score BM25 :",
            (
                round(
                    result.bm25_score,
                    6,
                )
                if result.bm25_score
                is not None
                else None
            ),
        )

        preview = (
            result.content
            .replace("\n", " ")
            [:400]
        )

        print(
            "Contenu :",
            preview,
        )

    expected_filename = (
        canonical_filename(
            question["expected_source"]
        )
    )

    matching_ranks = [
        result.rank
        for result in results
        if canonical_filename(
            result.source
        )
        == expected_filename
    ]

    print()
    print("=" * 100)
    print("VÉRIFICATION")
    print("=" * 100)

    if matching_ranks:
        print(
            "Source attendue retrouvée "
            "au rang final :",
            matching_ranks[0],
        )
    else:
        print(
            "Source attendue absente "
            "du Top 10 hybride."
        )


if __name__ == "__main__":
    main()