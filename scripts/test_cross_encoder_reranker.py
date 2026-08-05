from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from src.reranking import CrossEncoderReranker
from src.retrieval.hybrid_rrf_retriever import (
    HybridRRFRetriever,
)


# ============================================================
# Chemins du projet
# ============================================================

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


# ============================================================
# Configuration
# ============================================================

QUESTION_ID = "Q001"

HYBRID_CANDIDATE_K = 50
HYBRID_TOP_K = 20
FINAL_TOP_K = 5
RRF_CONSTANT = 60

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_BATCH_SIZE = 4
RERANKER_MAX_LENGTH = 512
RERANKER_DEVICE = "auto"


# ============================================================
# Fonctions utilitaires
# ============================================================

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_questions(
    path: Path,
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier d'évaluation ne contient pas d'en-tête."
            )

        for row in reader:
            questions.append(
                {
                    key: str(value or "").strip()
                    for key, value in row.items()
                }
            )

    if not questions:
        raise ValueError(
            "Aucune question trouvée dans le fichier d'évaluation."
        )

    return questions


def find_question_index(
    questions: list[dict[str, str]],
    question_id: str,
) -> int:
    for index, question in enumerate(
        questions
    ):
        if question.get("id") == question_id:
            return index

    raise KeyError(
        f"Question introuvable : {question_id}"
    )


def load_query_embeddings(
    path: Path,
) -> np.ndarray:
    embeddings = np.load(
        path,
        allow_pickle=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Le tableau des embeddings doit avoir deux dimensions."
        )

    if embeddings.shape[1] != 768:
        raise ValueError(
            "La dimension attendue des embeddings est 768. "
            f"Dimension trouvée : {embeddings.shape[1]}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs NaN ou infinies."
        )

    return embeddings


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    print("=" * 100)
    print("TEST DU CROSS-ENCODER RERANKER")
    print("=" * 100)

    require_file(EVALUATION_PATH)
    require_file(QUERY_EMBEDDINGS_PATH)

    questions = load_questions(
        EVALUATION_PATH
    )

    query_embeddings = load_query_embeddings(
        QUERY_EMBEDDINGS_PATH
    )

    if len(questions) != len(query_embeddings):
        raise ValueError(
            "Le nombre de questions ne correspond pas "
            "au nombre d'embeddings."
        )

    question_index = find_question_index(
        questions,
        QUESTION_ID,
    )

    question_record = questions[
        question_index
    ]

    question_text = question_record[
        "question"
    ]

    query_embedding = query_embeddings[
        question_index
    ]

    print("Question ID :", QUESTION_ID)
    print("Question :", question_text)
    print(
        "Source attendue :",
        question_record.get(
            "expected_source",
            "",
        ),
    )
    print(
        "Dimension embedding :",
        query_embedding.shape,
    )
    print()

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    hybrid_retriever = HybridRRFRetriever(
        candidate_k=HYBRID_CANDIDATE_K,
        final_top_k=HYBRID_TOP_K,
        rrf_constant=RRF_CONSTANT,
    )

    retrieval_start = time.perf_counter()

    hybrid_results = hybrid_retriever.search(
        question=question_text,
        query_embedding=query_embedding,
        candidate_k=HYBRID_CANDIDATE_K,
        final_top_k=HYBRID_TOP_K,
    )

    retrieval_time_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000

    print(
        "Candidats Hybrid RRF :",
        len(hybrid_results),
    )
    print(
        "Temps Hybrid RRF :",
        round(retrieval_time_ms, 2),
        "ms",
    )
    print()

    # --------------------------------------------------------
    # Cross-Encoder Reranking
    # --------------------------------------------------------

    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL,
        device=RERANKER_DEVICE,
        batch_size=RERANKER_BATCH_SIZE,
        max_length=RERANKER_MAX_LENGTH,
    )

    print("Modèle reranker :", reranker.model_name)
    print("Device :", reranker.device)
    print(
        "Modèle chargé avant appel :",
        reranker.is_loaded,
    )
    print()
    print(
        "Premier lancement : le modèle peut être "
        "téléchargé et chargé en mémoire."
    )
    print()

    reranking_start = time.perf_counter()

    reranked_results = reranker.rerank(
        question=question_text,
        candidates=hybrid_results,
        top_k=FINAL_TOP_K,
    )

    reranking_time_ms = (
        time.perf_counter()
        - reranking_start
    ) * 1000

    print(
        "Modèle chargé après appel :",
        reranker.is_loaded,
    )
    print(
        "Temps de reranking :",
        round(reranking_time_ms, 2),
        "ms",
    )
    print(
        "Temps total retrieval + reranking :",
        round(
            retrieval_time_ms
            + reranking_time_ms,
            2,
        ),
        "ms",
    )

    print()
    print("=" * 100)
    print("TOP 5 APRÈS RERANKING")
    print("=" * 100)

    for result in reranked_results:
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

    reranker.unload_model()

    print()
    print("=" * 100)
    print("TEST TERMINÉ")
    print("=" * 100)


if __name__ == "__main__":
    main()