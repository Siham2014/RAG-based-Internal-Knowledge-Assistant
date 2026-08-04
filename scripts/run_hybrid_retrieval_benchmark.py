from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.retrieval_metrics import (
    build_format_summaries,
    build_summary,
    detect_question_format,
    find_evidence_rank,
    find_source_rank,
    load_evaluation_questions,
    recall_at_k,
    reciprocal_rank,
    write_csv,
)
from src.retrieval.hybrid_rrf_retriever import (
    HybridRRFRetriever,
    HybridSearchResult,
)


# ============================================================
# Configuration
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "03_retrieval_benchmark"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "hybrid_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "hybrid_summary.csv"
)

BY_FORMAT_PATH = (
    RESULTS_DIR
    / "hybrid_by_format.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "hybrid_manifest.json"
)

RETRIEVAL_METHOD = "hybrid_rrf"

EXPECTED_NUMBER_OF_QUESTIONS = 30
EXPECTED_NUMBER_OF_CHUNKS = 1478
EXPECTED_EMBEDDING_DIMENSION = 768

CANDIDATE_K = 50
FINAL_TOP_K = 10
RRF_CONSTANT = 60

CHUNKING_CONFIGURATION = "fixed_1024"
EMBEDDING_MODEL = "intfloat/e5-base-v2"


# ============================================================
# Fonctions utilitaires
# ============================================================

def require_file(path: Path) -> None:
    """
    Vérifie qu'un fichier obligatoire existe.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_query_embeddings(
    path: Path,
) -> np.ndarray:
    """
    Charge les embeddings E5 des questions.
    """

    embeddings = np.load(
        path,
        allow_pickle=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        EXPECTED_NUMBER_OF_QUESTIONS,
        EXPECTED_EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Forme attendue : {expected_shape}. "
            f"Forme trouvée : {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs "
            "NaN ou infinies."
        )

    return embeddings


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Sauvegarde un dictionnaire en JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Évaluation d'une question
# ============================================================

def evaluate_question(
    retriever: HybridRRFRetriever,
    row: dict[str, str],
    query_embedding: np.ndarray,
) -> dict[str, Any]:
    """
    Évalue une question avec Hybrid RRF.
    """

    start_time = time.perf_counter()

    results: list[HybridSearchResult] = (
        retriever.search(
            question=row["question"],
            query_embedding=query_embedding,
            final_top_k=FINAL_TOP_K,
            candidate_k=CANDIDATE_K,
        )
    )

    retrieval_time_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    source_rank = find_source_rank(
        results=results,
        expected_source=row["expected_source"],
    )

    evidence_rank = find_evidence_rank(
        results=results,
        gold_evidence=row["gold_evidence"],
    )

    top_result = (
        results[0]
        if results
        else None
    )

    return {
        "retrieval_method": RETRIEVAL_METHOD,
        "id": row["id"],
        "question": row["question"],
        "format": detect_question_format(row),
        "expected_source": row["expected_source"],
        "gold_evidence": row["gold_evidence"],

        "source_rank": (
            source_rank
            if source_rank is not None
            else ""
        ),
        "evidence_rank": (
            evidence_rank
            if evidence_rank is not None
            else ""
        ),

        "source_recall_at_1": recall_at_k(
            source_rank,
            1,
        ),
        "source_recall_at_3": recall_at_k(
            source_rank,
            3,
        ),
        "source_recall_at_5": recall_at_k(
            source_rank,
            5,
        ),
        "source_recall_at_10": recall_at_k(
            source_rank,
            10,
        ),
        "source_reciprocal_rank": round(
            reciprocal_rank(
                source_rank
            ),
            6,
        ),

        "evidence_recall_at_1": recall_at_k(
            evidence_rank,
            1,
        ),
        "evidence_recall_at_3": recall_at_k(
            evidence_rank,
            3,
        ),
        "evidence_recall_at_5": recall_at_k(
            evidence_rank,
            5,
        ),
        "evidence_recall_at_10": recall_at_k(
            evidence_rank,
            10,
        ),
        "evidence_reciprocal_rank": round(
            reciprocal_rank(
                evidence_rank
            ),
            6,
        ),

        "top_1_chunk_id": (
            top_result.chunk_id
            if top_result
            else ""
        ),
        "top_1_source": (
            top_result.source
            if top_result
            else ""
        ),
        "top_1_rrf_score": (
            round(
                top_result.rrf_score,
                8,
            )
            if top_result
            else ""
        ),
        "top_1_vector_rank": (
            top_result.vector_rank
            if top_result
            and top_result.vector_rank is not None
            else ""
        ),
        "top_1_bm25_rank": (
            top_result.bm25_rank
            if top_result
            and top_result.bm25_rank is not None
            else ""
        ),
        "top_1_cosine_similarity": (
            round(
                top_result.cosine_similarity,
                6,
            )
            if top_result
            and top_result.cosine_similarity is not None
            else ""
        ),
        "top_1_bm25_score": (
            round(
                top_result.bm25_score,
                6,
            )
            if top_result
            and top_result.bm25_score is not None
            else ""
        ),

        "number_of_results": len(results),
        "retrieval_time_ms": round(
            retrieval_time_ms,
            4,
        ),
    }


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    """
    Lance le benchmark Hybrid RRF sur 30 questions.
    """

    print("=" * 100)
    print(
        "BENCHMARK HYBRID RETRIEVAL — PGVECTOR + BM25 + RRF"
    )
    print("=" * 100)

    require_file(
        EVALUATION_PATH
    )

    require_file(
        QUERY_EMBEDDINGS_PATH
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_evaluation_questions(
        EVALUATION_PATH
    )

    query_embeddings = (
        load_query_embeddings(
            QUERY_EMBEDDINGS_PATH
        )
    )

    if len(questions) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"questions attendues, "
            f"{len(questions)} trouvées."
        )

    if len(questions) != len(
        query_embeddings
    ):
        raise ValueError(
            "Le nombre de questions ne correspond pas "
            "au nombre d'embeddings."
        )

    retriever = HybridRRFRetriever(
        candidate_k=CANDIDATE_K,
        final_top_k=FINAL_TOP_K,
        rrf_constant=RRF_CONSTANT,
    )

    vector_chunks = (
        retriever.count_vector_chunks()
    )

    bm25_chunks = (
        retriever.count_bm25_chunks()
    )

    if vector_chunks != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_CHUNKS} chunks "
            f"attendus dans pgvector, "
            f"{vector_chunks} trouvés."
        )

    if bm25_chunks != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_CHUNKS} chunks "
            f"attendus dans BM25, "
            f"{bm25_chunks} trouvés."
        )

    print(
        "Méthode :",
        RETRIEVAL_METHOD,
    )

    print(
        "Chunks pgvector :",
        vector_chunks,
    )

    print(
        "Chunks BM25 :",
        bm25_chunks,
    )

    print(
        "Questions :",
        len(questions),
    )

    print(
        "Embeddings :",
        query_embeddings.shape,
    )

    print(
        "Candidats pgvector :",
        CANDIDATE_K,
    )

    print(
        "Candidats BM25 :",
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

    print()

    benchmark_start = (
        time.perf_counter()
    )

    detail_records: list[
        dict[str, Any]
    ] = []

    for index, (
        question,
        query_embedding,
    ) in enumerate(
        zip(
            questions,
            query_embeddings,
        ),
        start=1,
    ):

        record = evaluate_question(
            retriever=retriever,
            row=question,
            query_embedding=query_embedding,
        )

        detail_records.append(
            record
        )

        evidence_rank_display = (
            record["evidence_rank"]
            if record["evidence_rank"] != ""
            else "absente"
        )

        source_rank_display = (
            record["source_rank"]
            if record["source_rank"] != ""
            else "absente"
        )

        print(
            f"[{index:02d}/{len(questions)}] "
            f"{record['id']} | "
            f"Evidence rank: "
            f"{evidence_rank_display} | "
            f"Source rank: "
            f"{source_rank_display} | "
            f"Résultats: "
            f"{record['number_of_results']} | "
            f"{record['retrieval_time_ms']:.2f} ms"
        )

    total_benchmark_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    summary_record = build_summary(
        records=detail_records,
        retrieval_method=(
            RETRIEVAL_METHOD
        ),
        number_of_indexed_chunks=(
            EXPECTED_NUMBER_OF_CHUNKS
        ),
        total_benchmark_seconds=(
            total_benchmark_seconds
        ),
        extra_fields={
            "vector_engine": (
                "PostgreSQL pgvector"
            ),
            "lexical_engine": (
                "SQLite FTS5 BM25"
            ),
            "fusion_method": (
                "Reciprocal Rank Fusion"
            ),
            "candidate_k_per_engine": (
                CANDIDATE_K
            ),
            "final_top_k": (
                FINAL_TOP_K
            ),
            "rrf_constant": (
                RRF_CONSTANT
            ),
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "chunking_configuration": (
                CHUNKING_CONFIGURATION
            ),
        },
    )

    format_records = (
        build_format_summaries(
            records=detail_records,
            retrieval_method=(
                RETRIEVAL_METHOD
            ),
        )
    )

    write_csv(
        DETAILS_PATH,
        detail_records,
    )

    write_csv(
        SUMMARY_PATH,
        [summary_record],
    )

    write_csv(
        BY_FORMAT_PATH,
        format_records,
    )

    manifest = {
        "benchmark_name": (
            "hybrid_retrieval_rrf"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "retrieval_method": (
            RETRIEVAL_METHOD
        ),
        "vector_engine": (
            "PostgreSQL pgvector"
        ),
        "lexical_engine": (
            "SQLite FTS5 BM25"
        ),
        "fusion_method": (
            "Reciprocal Rank Fusion"
        ),
        "rrf_constant": (
            RRF_CONSTANT
        ),
        "candidate_k_per_engine": (
            CANDIDATE_K
        ),
        "final_top_k": (
            FINAL_TOP_K
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "embedding_dimension": (
            EXPECTED_EMBEDDING_DIMENSION
        ),
        "chunking_configuration": (
            CHUNKING_CONFIGURATION
        ),
        "number_of_indexed_chunks": (
            EXPECTED_NUMBER_OF_CHUNKS
        ),
        "number_of_questions": (
            len(questions)
        ),
        "input_files": {
            "evaluation_questions": str(
                EVALUATION_PATH
            ),
            "query_embeddings": str(
                QUERY_EMBEDDINGS_PATH
            ),
        },
        "output_files": {
            "summary": str(
                SUMMARY_PATH
            ),
            "details": str(
                DETAILS_PATH
            ),
            "by_format": str(
                BY_FORMAT_PATH
            ),
        },
        "summary": summary_record,
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print()
    print("=" * 100)
    print(
        "BENCHMARK HYBRID RRF TERMINÉ"
    )
    print("=" * 100)

    print(
        "Evidence Recall@1 :",
        summary_record[
            "evidence_recall_at_1"
        ],
    )

    print(
        "Evidence Recall@3 :",
        summary_record[
            "evidence_recall_at_3"
        ],
    )

    print(
        "Evidence Recall@5 :",
        summary_record[
            "evidence_recall_at_5"
        ],
    )

    print(
        "Evidence Recall@10 :",
        summary_record[
            "evidence_recall_at_10"
        ],
    )

    print(
        "Evidence MRR :",
        summary_record[
            "evidence_mrr"
        ],
    )

    print(
        "Source Recall@5 :",
        summary_record[
            "source_recall_at_5"
        ],
    )

    print(
        "Temps moyen :",
        summary_record[
            "mean_retrieval_time_ms"
        ],
        "ms",
    )

    print(
        "P95 :",
        summary_record[
            "p95_retrieval_time_ms"
        ],
        "ms",
    )

    print()
    print(
        "Résumé :",
        SUMMARY_PATH,
    )

    print(
        "Détails :",
        DETAILS_PATH,
    )

    print(
        "Par format :",
        BY_FORMAT_PATH,
    )

    print(
        "Manifeste :",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()