from __future__ import annotations

import csv
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

HYDE_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "query_rewriting"
    / "hyde_e5_query_embeddings.npy"
)

HYDE_DOCUMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "query_rewriting"
    / "hyde_documents.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "05_query_rewriting_benchmark"
)

SUMMARY_PATH = RESULTS_DIR / "hyde_hybrid_summary.csv"
DETAILS_PATH = RESULTS_DIR / "hyde_hybrid_details.csv"
BY_FORMAT_PATH = RESULTS_DIR / "hyde_hybrid_by_format.csv"
CANDIDATES_PATH = RESULTS_DIR / "hyde_top20_candidates.jsonl"
MANIFEST_PATH = RESULTS_DIR / "hyde_hybrid_manifest.json"

RETRIEVAL_METHOD = "hyde_hybrid_rrf"

EXPECTED_NUMBER_OF_QUESTIONS = 30
EXPECTED_NUMBER_OF_CHUNKS = 1478
EXPECTED_EMBEDDING_DIMENSION = 768

CANDIDATE_K_PER_ENGINE = 50
FINAL_TOP_K_FOR_METRICS = 10
EXPORT_TOP_K_FOR_RERANKING = 20
RRF_CONSTANT = 60

GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
CHUNKING_CONFIGURATION = "fixed_1024"


# ============================================================
# Chargement
# ============================================================

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_hyde_embeddings(path: Path) -> np.ndarray:
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
            "Les embeddings HyDE contiennent "
            "des valeurs NaN ou infinies."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Au moins un embedding HyDE "
            "possède une norme nulle."
        )

    embeddings = (
        embeddings
        / norms[:, None]
    )

    return embeddings.astype(
        np.float32,
        copy=False,
    )


def load_hyde_documents(
    path: Path,
) -> dict[str, dict[str, str]]:
    """
    Charge les passages HyDE générés sur Kaggle.
    """

    documents: dict[
        str,
        dict[str, str]
    ] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le CSV HyDE ne contient pas d'en-tête."
            )

        required_columns = {
            "id",
            "question",
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Colonnes manquantes dans le CSV HyDE : "
                + ", ".join(sorted(missing))
            )

        document_column = None

        for candidate_column in (
            "hyde_document",
            "hypothetical_document",
        ):
            if candidate_column in reader.fieldnames:
                document_column = candidate_column
                break

        if document_column is None:
            raise ValueError(
                "Aucune colonne HyDE trouvée. "
                "Colonnes attendues : hyde_document "
                "ou hypothetical_document."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            question_id = str(
                row.get("id", "")
            ).strip()

            question = str(
                row.get("question", "")
            ).strip()

            hyde_document = str(
                row.get(document_column, "")
            ).strip()

            if not question_id:
                raise ValueError(
                    f"ID vide à la ligne {row_number}."
                )

            if not question:
                raise ValueError(
                    f"Question vide à la ligne {row_number}."
                )

            if not hyde_document:
                raise ValueError(
                    f"Document HyDE vide à la ligne "
                    f"{row_number}."
                )

            documents[question_id] = {
                "question": question,
                "hyde_document": hyde_document,
            }

    return documents


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
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
# Évaluation
# ============================================================

def evaluate_question(
    retriever: HybridRRFRetriever,
    question_row: dict[str, str],
    hyde_embedding: np.ndarray,
) -> tuple[
    dict[str, Any],
    list[HybridSearchResult],
]:
    """
    Branche vectorielle : embedding HyDE.
    Branche lexicale : question originale.
    """

    start_time = time.perf_counter()

    results = retriever.search(
        question=question_row["question"],
        query_embedding=hyde_embedding,
        candidate_k=CANDIDATE_K_PER_ENGINE,
        final_top_k=EXPORT_TOP_K_FOR_RERANKING,
    )

    retrieval_time_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    metric_results = results[
        :FINAL_TOP_K_FOR_METRICS
    ]

    source_rank = find_source_rank(
        metric_results,
        question_row["expected_source"],
    )

    evidence_rank = find_evidence_rank(
        metric_results,
        question_row["gold_evidence"],
    )

    top_result = (
        metric_results[0]
        if metric_results
        else None
    )

    detail = {
        "retrieval_method": RETRIEVAL_METHOD,
        "id": question_row["id"],
        "question": question_row["question"],
        "format": detect_question_format(
            question_row
        ),
        "expected_source": (
            question_row["expected_source"]
        ),
        "gold_evidence": (
            question_row["gold_evidence"]
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
            reciprocal_rank(source_rank),
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
            reciprocal_rank(evidence_rank),
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
            round(top_result.rrf_score, 8)
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
        "retrieval_time_ms": round(
            retrieval_time_ms,
            4,
        ),
    }

    return detail, results


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    print("=" * 100)
    print(
        "BENCHMARK QUERY REWRITING — "
        "HYDE + HYBRID RRF"
    )
    print("=" * 100)

    require_file(EVALUATION_PATH)
    require_file(HYDE_EMBEDDINGS_PATH)
    require_file(HYDE_DOCUMENTS_PATH)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_evaluation_questions(
        EVALUATION_PATH
    )

    hyde_embeddings = load_hyde_embeddings(
        HYDE_EMBEDDINGS_PATH
    )

    hyde_documents = load_hyde_documents(
        HYDE_DOCUMENTS_PATH
    )

    if len(questions) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"questions attendues, "
            f"{len(questions)} trouvées."
        )

    if len(hyde_documents) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"documents HyDE attendus, "
            f"{len(hyde_documents)} trouvés."
        )

    retriever = HybridRRFRetriever(
        candidate_k=(
            CANDIDATE_K_PER_ENGINE
        ),
        final_top_k=(
            EXPORT_TOP_K_FOR_RERANKING
        ),
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
            f"pgvector contient {vector_chunks} "
            f"chunks au lieu de "
            f"{EXPECTED_NUMBER_OF_CHUNKS}."
        )

    if bm25_chunks != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise ValueError(
            f"BM25 contient {bm25_chunks} "
            f"chunks au lieu de "
            f"{EXPECTED_NUMBER_OF_CHUNKS}."
        )

    print(
        "Générateur HyDE :",
        GENERATOR_MODEL,
    )
    print(
        "Modèle embedding :",
        EMBEDDING_MODEL,
    )
    print(
        "Embeddings HyDE :",
        hyde_embeddings.shape,
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
        "Top K métriques :",
        FINAL_TOP_K_FOR_METRICS,
    )
    print(
        "Top K export reranking :",
        EXPORT_TOP_K_FOR_RERANKING,
    )
    print()

    benchmark_start = (
        time.perf_counter()
    )

    detail_records: list[
        dict[str, Any]
    ] = []

    with CANDIDATES_PATH.open(
        "w",
        encoding="utf-8",
    ) as candidate_file:

        for index, (
            question,
            hyde_embedding,
        ) in enumerate(
            zip(
                questions,
                hyde_embeddings,
            ),
            start=1,
        ):
            question_id = question["id"]

            if question_id not in hyde_documents:
                raise KeyError(
                    f"Document HyDE absent pour "
                    f"{question_id}."
                )

            detail, results = evaluate_question(
                retriever=retriever,
                question_row=question,
                hyde_embedding=hyde_embedding,
            )

            detail_records.append(
                detail
            )

            export_record = {
                "id": question_id,
                "question": question["question"],
                "hyde_document": (
                    hyde_documents[
                        question_id
                    ]["hyde_document"]
                ),
                "format": (
                    detect_question_format(
                        question
                    )
                ),
                "expected_source": (
                    question["expected_source"]
                ),
                "gold_evidence": (
                    question["gold_evidence"]
                ),
                "retrieval_time_ms": (
                    detail["retrieval_time_ms"]
                ),
                "candidates": [],
            }

            for result in results:
                export_record[
                    "candidates"
                ].append(
                    {
                        "hybrid_rank": (
                            result.rank
                        ),
                        "chunk_position": (
                            result.chunk_position
                        ),
                        "chunk_id": (
                            result.chunk_id
                        ),
                        "content": (
                            result.content
                        ),
                        "source": (
                            result.source
                        ),
                        "document_format": (
                            result.document_format
                        ),
                        "page_number": (
                            result.page_number
                        ),
                        "rrf_score": round(
                            result.rrf_score,
                            10,
                        ),
                        "vector_rank": (
                            result.vector_rank
                        ),
                        "bm25_rank": (
                            result.bm25_rank
                        ),
                        "cosine_similarity": (
                            result.cosine_similarity
                        ),
                        "bm25_score": (
                            result.bm25_score
                        ),
                    }
                )

            candidate_file.write(
                json.dumps(
                    export_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            evidence_display = (
                detail["evidence_rank"]
                if detail["evidence_rank"] != ""
                else "absente"
            )

            print(
                f"[{index:02d}/"
                f"{len(questions)}] "
                f"{question_id} | "
                f"Evidence rank: "
                f"{evidence_display} | "
                f"{detail['retrieval_time_ms']:.2f} ms"
            )

    total_benchmark_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    summary_record = build_summary(
        records=detail_records,
        retrieval_method=RETRIEVAL_METHOD,
        number_of_indexed_chunks=(
            EXPECTED_NUMBER_OF_CHUNKS
        ),
        total_benchmark_seconds=(
            total_benchmark_seconds
        ),
        extra_fields={
            "query_rewriting": "HyDE",
            "generator_model": (
                GENERATOR_MODEL
            ),
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "chunking_configuration": (
                CHUNKING_CONFIGURATION
            ),
            "vector_candidate_k": (
                CANDIDATE_K_PER_ENGINE
            ),
            "bm25_candidate_k": (
                CANDIDATE_K_PER_ENGINE
            ),
            "metric_top_k": (
                FINAL_TOP_K_FOR_METRICS
            ),
            "export_top_k": (
                EXPORT_TOP_K_FOR_RERANKING
            ),
            "rrf_constant": (
                RRF_CONSTANT
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
            "query_rewriting_hyde"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "query_rewriting": "HyDE",
        "generator_model": (
            GENERATOR_MODEL
        ),
        "embedding_model": (
            EMBEDDING_MODEL
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
        "number_of_questions": (
            len(questions)
        ),
        "number_of_indexed_chunks": (
            EXPECTED_NUMBER_OF_CHUNKS
        ),
        "rrf_constant": RRF_CONSTANT,
        "vector_candidate_k": (
            CANDIDATE_K_PER_ENGINE
        ),
        "bm25_candidate_k": (
            CANDIDATE_K_PER_ENGINE
        ),
        "metric_top_k": (
            FINAL_TOP_K_FOR_METRICS
        ),
        "export_top_k": (
            EXPORT_TOP_K_FOR_RERANKING
        ),
        "input_files": {
            "evaluation": str(
                EVALUATION_PATH
            ),
            "hyde_embeddings": str(
                HYDE_EMBEDDINGS_PATH
            ),
            "hyde_documents": str(
                HYDE_DOCUMENTS_PATH
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
            "reranking_candidates": str(
                CANDIDATES_PATH
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
        "BENCHMARK HYDE + HYBRID RRF TERMINÉ"
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
        "Temps moyen :",
        summary_record[
            "mean_retrieval_time_ms"
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
        "Top 20 reranking :",
        CANDIDATES_PATH,
    )
    print(
        "Manifeste :",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()