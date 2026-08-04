from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25SearchResult,
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "03_retrieval_benchmark"
)

DETAILS_PATH = (
    RESULTS_DIR
    / "bm25_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "bm25_summary.csv"
)

BY_FORMAT_PATH = (
    RESULTS_DIR
    / "bm25_by_format.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "bm25_manifest.json"
)

RETRIEVAL_METHOD = "sqlite_fts5_bm25"

EXPECTED_NUMBER_OF_QUESTIONS = 30
EXPECTED_NUMBER_OF_CHUNKS = 1478

TOP_K = 10

CHUNKING_CONFIGURATION = "fixed_1024"

DATABASE_ENGINE = "SQLite FTS5"


# ============================================================
# Fonctions du benchmark
# ============================================================

def require_file(
    path: Path,
) -> None:
    """
    Vérifie qu'un fichier nécessaire existe.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def evaluate_question(
    retriever: BM25Retriever,
    row: dict[str, str],
) -> dict[str, Any]:
    """
    Exécute une question avec BM25 et calcule
    ses rangs et métriques.
    """

    start_time = time.perf_counter()

    results: list[BM25SearchResult] = (
        retriever.search(
            question=row["question"],
            top_k=TOP_K,
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
        "format": detect_question_format(
            row
        ),
        "expected_source": (
            row["expected_source"]
        ),
        "gold_evidence": (
            row["gold_evidence"]
        ),

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
        "top_1_bm25_score": (
            round(
                top_result.bm25_score,
                6,
            )
            if top_result
            else ""
        ),
        "number_of_results": len(
            results
        ),
        "retrieval_time_ms": round(
            retrieval_time_ms,
            4,
        ),
    }


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    Sauvegarde un dictionnaire au format JSON.
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
# Programme principal
# ============================================================

def main() -> None:
    """
    Lance le benchmark BM25 sur les 30 questions.
    """

    print("=" * 100)
    print(
        "BENCHMARK LEXICAL RETRIEVAL — SQLITE FTS5 / BM25"
    )
    print("=" * 100)

    require_file(
        EVALUATION_PATH
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_evaluation_questions(
        EVALUATION_PATH
    )

    if len(questions) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"questions attendues, "
            f"{len(questions)} trouvées."
        )

    retriever = BM25Retriever(
        default_top_k=TOP_K
    )

    indexed_chunks = (
        retriever.count_indexed_chunks()
    )

    if indexed_chunks != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_CHUNKS} "
            f"chunks attendus, "
            f"{indexed_chunks} trouvés."
        )

    print(
        "Méthode :",
        RETRIEVAL_METHOD,
    )

    print(
        "Moteur :",
        DATABASE_ENGINE,
    )

    print(
        "Chunks indexés :",
        indexed_chunks,
    )

    print(
        "Questions :",
        len(questions),
    )

    print(
        "Top K :",
        TOP_K,
    )

    print()

    benchmark_start = (
        time.perf_counter()
    )

    detail_records: list[
        dict[str, Any]
    ] = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        record = evaluate_question(
            retriever=retriever,
            row=question,
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
            indexed_chunks
        ),
        total_benchmark_seconds=(
            total_benchmark_seconds
        ),
        extra_fields={
            "database_engine": (
                DATABASE_ENGINE
            ),
            "chunking_configuration": (
                CHUNKING_CONFIGURATION
            ),
            "top_k": TOP_K,
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
            "lexical_retrieval_bm25"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "retrieval_method": (
            RETRIEVAL_METHOD
        ),
        "database_engine": (
            DATABASE_ENGINE
        ),
        "search_mode": "lexical",
        "ranking_function": "bm25",
        "tokenizer": (
            "unicode61 remove_diacritics 2"
        ),
        "query_operator": "OR",
        "chunking_configuration": (
            CHUNKING_CONFIGURATION
        ),
        "number_of_indexed_chunks": (
            indexed_chunks
        ),
        "number_of_questions": (
            len(questions)
        ),
        "top_k": TOP_K,
        "input_files": {
            "evaluation_questions": str(
                EVALUATION_PATH
            ),
            "bm25_database": str(
                retriever.database_path
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
    print("BENCHMARK BM25 TERMINÉ")
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