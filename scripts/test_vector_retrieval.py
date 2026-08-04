from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from src.retrieval.vector_retriever import (
    PgVectorRetriever,
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

TOP_K = 10


def load_evaluation_questions(
    path: Path,
) -> list[dict[str, str]]:
    """
    Charge les questions d'évaluation depuis le CSV.
    """

    questions: list[dict[str, str]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "id",
            "question",
            "expected_source",
        }

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier CSV ne contient pas d'en-tête."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Colonnes manquantes dans le CSV : "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            question_id = str(
                row.get("id", "")
            ).strip()

            question_text = str(
                row.get("question", "")
            ).strip()

            expected_source = str(
                row.get("expected_source", "")
            ).strip()

            if not question_id:
                raise ValueError(
                    f"ID vide à la ligne {row_number}."
                )

            if not question_text:
                raise ValueError(
                    f"Question vide à la ligne {row_number}."
                )

            if not expected_source:
                raise ValueError(
                    f"Source attendue vide à la ligne "
                    f"{row_number}."
                )

            questions.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "expected_source": expected_source,
                }
            )

    return questions


def canonical_filename(
    source: str,
) -> str:
    """
    Retourne uniquement le nom du fichier,
    quel que soit le séparateur utilisé.
    """

    normalized = str(source).replace(
        "\\",
        "/",
    )

    return normalized.split("/")[-1]


def main() -> None:
    """
    Teste une première recherche vectorielle pgvector.
    """

    if not EVALUATION_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {EVALUATION_PATH}"
        )

    if not QUERY_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{QUERY_EMBEDDINGS_PATH}"
        )

    evaluation_questions = (
        load_evaluation_questions(
            EVALUATION_PATH
        )
    )

    query_embeddings = np.load(
        QUERY_EMBEDDINGS_PATH,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    if len(evaluation_questions) != len(
        query_embeddings
    ):
        raise ValueError(
            "Le nombre de questions ne correspond pas "
            "au nombre d'embeddings : "
            f"{len(evaluation_questions)} questions, "
            f"{len(query_embeddings)} embeddings."
        )

    if query_embeddings.shape != (
        30,
        768,
    ):
        raise ValueError(
            "Forme inattendue pour les embeddings "
            f"des questions : {query_embeddings.shape}. "
            "Forme attendue : (30, 768)."
        )

    first_question = (
        evaluation_questions[0]
    )

    question_id = first_question["id"]
    question_text = first_question["question"]
    expected_source = (
        first_question["expected_source"]
    )

    query_embedding = (
        query_embeddings[0]
    )

    retriever = PgVectorRetriever(
        default_top_k=TOP_K
    )

    indexed_chunks = (
        retriever.count_indexed_chunks()
    )

    print("=" * 100)
    print(
        "TEST DU VECTOR RETRIEVAL AVEC PGVECTOR"
    )
    print("=" * 100)

    print(
        "Chunks indexés :",
        indexed_chunks,
    )

    print(
        "Questions chargées :",
        len(evaluation_questions),
    )

    print(
        "Embeddings questions :",
        query_embeddings.shape,
    )

    print(
        "Question ID :",
        question_id,
    )

    print(
        "Question :",
        question_text,
    )

    print(
        "Source attendue :",
        expected_source,
    )

    print(
        "Dimension embedding :",
        query_embedding.shape,
    )

    start_time = time.perf_counter()

    results = retriever.search_by_embedding(
        query_embedding=query_embedding,
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
    print("TOP 10")
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
            "Similarité cosinus :",
            round(
                result.cosine_similarity,
                6,
            ),
        )

        print(
            "Distance cosinus :",
            round(
                result.cosine_distance,
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
            expected_source
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
            "Source attendue retrouvée au rang :",
            matching_ranks[0],
        )
    else:
        print(
            "Source attendue absente du Top 10."
        )


if __name__ == "__main__":
    main()