from __future__ import annotations

import csv
import time
from pathlib import Path

from src.retrieval.bm25_retriever import (
    BM25Retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_30_questions.csv"
)

TOP_K = 10


def load_first_question(
    path: Path,
) -> dict[str, str]:
    """
    Charge la première question du CSV.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        first_row = next(
            reader,
            None,
        )

    if first_row is None:
        raise ValueError(
            "Le fichier d'évaluation est vide."
        )

    return {
        key: str(
            value or ""
        ).strip()
        for key, value in first_row.items()
        if key is not None
    }


def canonical_filename(
    source: str,
) -> str:
    """
    Retourne le nom du fichier.
    """

    return (
        str(source)
        .replace("\\", "/")
        .split("/")[-1]
        .lower()
    )


def main() -> None:
    """
    Teste BM25 sur Q001.
    """

    question = load_first_question(
        EVALUATION_PATH
    )

    retriever = BM25Retriever(
        default_top_k=TOP_K
    )

    print("=" * 100)
    print(
        "TEST DU RETRIEVAL BM25 — SQLITE FTS5"
    )
    print("=" * 100)

    print(
        "Chunks indexés :",
        retriever.count_indexed_chunks(),
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

    start_time = time.perf_counter()

    results = retriever.search(
        question=question["question"],
        top_k=TOP_K,
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    print()
    print(
        f"Recherche terminée en "
        f"{elapsed_ms:.3f} ms"
    )

    print()
    print("=" * 100)
    print("TOP 10 BM25")
    print("=" * 100)

    for result in results:

        print()
        print("-" * 100)

        print(
            "Rang :",
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
            "Score BM25 :",
            round(
                result.bm25_score,
                6,
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
            question[
                "expected_source"
            ]
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
            "au rang :",
            matching_ranks[0],
        )
    else:
        print(
            "Source attendue absente "
            "du Top 10."
        )


if __name__ == "__main__":
    main()