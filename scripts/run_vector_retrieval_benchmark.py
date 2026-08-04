from __future__ import annotations

import csv
import json
import math
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.vector_retriever import (
    PgVectorRetriever,
    VectorSearchResult,
)


# ============================================================
# Configuration générale
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
    / "pgvector_details.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "pgvector_summary.csv"
)

BY_FORMAT_PATH = (
    RESULTS_DIR
    / "pgvector_by_format.csv"
)

MANIFEST_PATH = (
    RESULTS_DIR
    / "pgvector_manifest.json"
)

TOP_K = 10
EXPECTED_NUMBER_OF_QUESTIONS = 30
EXPECTED_EMBEDDING_DIMENSION = 768

RETRIEVAL_METHOD = "pgvector_exact"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
CHUNKING_CONFIGURATION = "fixed_1024"


# ============================================================
# Chargement du jeu d'évaluation
# ============================================================

def require_file(path: Path) -> None:
    """
    Vérifie qu'un fichier obligatoire existe.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_evaluation_questions(
    path: Path,
) -> list[dict[str, str]]:
    """
    Charge les 30 questions depuis le fichier CSV.
    """

    questions: list[dict[str, str]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier CSV ne possède pas d'en-tête."
            )

        required_columns = {
            "id",
            "question",
            "expected_source",
            "gold_evidence",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Colonnes manquantes : "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):

            normalized_row = {
                key: str(value or "").strip()
                for key, value in row.items()
                if key is not None
            }

            for required_column in required_columns:
                if not normalized_row.get(required_column):
                    raise ValueError(
                        f"Valeur vide pour "
                        f"'{required_column}' "
                        f"à la ligne {row_number}."
                    )

            questions.append(
                normalized_row
            )

    return questions


def load_query_embeddings(
    path: Path,
) -> np.ndarray:
    """
    Charge les embeddings E5 des 30 questions.
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
            "Les embeddings de questions contiennent "
            "des valeurs NaN ou infinies."
        )

    return embeddings


# ============================================================
# Normalisation des textes et des sources
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalise un texte pour comparer les preuves.

    La normalisation ignore :
    - les différences majuscules/minuscules ;
    - les espaces multiples ;
    - les retours à la ligne ;
    - les variantes Unicode.
    """

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    normalized = normalized.lower()

    normalized = " ".join(
        normalized.split()
    )

    return normalized.strip()


def canonical_filename(source: str) -> str:
    """
    Retourne uniquement le nom final du fichier.
    """

    normalized = str(source).replace(
        "\\",
        "/",
    )

    return normalized.split("/")[-1].lower()


def detect_question_format(
    row: dict[str, str],
) -> str:
    """
    Détermine le format documentaire attendu.

    Utilise d'abord la colonne format si elle existe,
    sinon déduit le format depuis expected_source.
    """

    explicit_format = (
        row.get("format", "")
        or row.get("document_format", "")
    ).strip().lower()

    if explicit_format in {
        "html",
        "markdown",
        "pdf",
    }:
        return explicit_format

    source = row["expected_source"].lower()

    if source.endswith(".md"):
        return "markdown"

    if source.endswith(".html"):
        return "html"

    if source.endswith(".pdf"):
        return "pdf"

    return "unknown"


# ============================================================
# Calcul des rangs
# ============================================================

def find_source_rank(
    results: list[VectorSearchResult],
    expected_source: str,
) -> int | None:
    """
    Retourne le premier rang correspondant à la source.
    """

    expected_filename = canonical_filename(
        expected_source
    )

    for result in results:
        result_filename = canonical_filename(
            result.source
        )

        if result_filename == expected_filename:
            return result.rank

    return None


def find_evidence_rank(
    results: list[VectorSearchResult],
    gold_evidence: str,
) -> int | None:
    """
    Retourne le premier rang contenant la preuve exacte
    après normalisation.
    """

    normalized_evidence = normalize_text(
        gold_evidence
    )

    for result in results:

        normalized_content = normalize_text(
            result.content
        )

        if normalized_evidence in normalized_content:
            return result.rank

    return None


def recall_at_k(
    rank: int | None,
    k: int,
) -> int:
    """
    Retourne 1 si le rang est inférieur ou égal à k.
    """

    if rank is None:
        return 0

    return int(rank <= k)


def reciprocal_rank(
    rank: int | None,
) -> float:
    """
    Calcule le reciprocal rank.
    """

    if rank is None:
        return 0.0

    return 1.0 / rank


# ============================================================
# Statistiques
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calcule un percentile avec interpolation linéaire.
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (
        len(sorted_values) - 1
    ) * percentile_value

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(
            sorted_values[lower_index]
        )

    lower_value = sorted_values[
        lower_index
    ]

    upper_value = sorted_values[
        upper_index
    ]

    fraction = position - lower_index

    return float(
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def mean_field(
    records: list[dict[str, Any]],
    field_name: str,
) -> float:
    """
    Calcule la moyenne d'un champ numérique.
    """

    if not records:
        return 0.0

    return float(
        statistics.fmean(
            float(record[field_name])
            for record in records
        )
    )


def build_summary(
    records: list[dict[str, Any]],
    indexed_chunks: int,
    total_benchmark_seconds: float,
) -> dict[str, Any]:
    """
    Construit le résumé global du benchmark.
    """

    retrieval_times = [
        float(record["retrieval_time_ms"])
        for record in records
    ]

    return {
        "retrieval_method": RETRIEVAL_METHOD,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": (
            EXPECTED_EMBEDDING_DIMENSION
        ),
        "chunking_configuration": (
            CHUNKING_CONFIGURATION
        ),
        "number_of_indexed_chunks": indexed_chunks,
        "number_of_questions": len(records),

        "evidence_recall_at_1": round(
            mean_field(
                records,
                "evidence_recall_at_1",
            ),
            4,
        ),
        "evidence_recall_at_3": round(
            mean_field(
                records,
                "evidence_recall_at_3",
            ),
            4,
        ),
        "evidence_recall_at_5": round(
            mean_field(
                records,
                "evidence_recall_at_5",
            ),
            4,
        ),
        "evidence_recall_at_10": round(
            mean_field(
                records,
                "evidence_recall_at_10",
            ),
            4,
        ),
        "evidence_mrr": round(
            mean_field(
                records,
                "evidence_reciprocal_rank",
            ),
            4,
        ),

        "source_recall_at_1": round(
            mean_field(
                records,
                "source_recall_at_1",
            ),
            4,
        ),
        "source_recall_at_3": round(
            mean_field(
                records,
                "source_recall_at_3",
            ),
            4,
        ),
        "source_recall_at_5": round(
            mean_field(
                records,
                "source_recall_at_5",
            ),
            4,
        ),
        "source_recall_at_10": round(
            mean_field(
                records,
                "source_recall_at_10",
            ),
            4,
        ),
        "source_mrr": round(
            mean_field(
                records,
                "source_reciprocal_rank",
            ),
            4,
        ),

        "mean_retrieval_time_ms": round(
            statistics.fmean(
                retrieval_times
            ),
            4,
        ),
        "median_retrieval_time_ms": round(
            statistics.median(
                retrieval_times
            ),
            4,
        ),
        "p95_retrieval_time_ms": round(
            percentile(
                retrieval_times,
                0.95,
            ),
            4,
        ),
        "minimum_retrieval_time_ms": round(
            min(retrieval_times),
            4,
        ),
        "maximum_retrieval_time_ms": round(
            max(retrieval_times),
            4,
        ),
        "total_benchmark_time_seconds": round(
            total_benchmark_seconds,
            4,
        ),
    }


def build_format_summaries(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Produit les métriques séparées par format.
    """

    formats = sorted(
        {
            str(record["format"])
            for record in records
        }
    )

    format_summaries: list[
        dict[str, Any]
    ] = []

    for document_format in formats:

        format_records = [
            record
            for record in records
            if record["format"]
            == document_format
        ]

        retrieval_times = [
            float(
                record[
                    "retrieval_time_ms"
                ]
            )
            for record in format_records
        ]

        format_summaries.append(
            {
                "retrieval_method": (
                    RETRIEVAL_METHOD
                ),
                "format": document_format,
                "number_of_questions": len(
                    format_records
                ),

                "evidence_recall_at_1": round(
                    mean_field(
                        format_records,
                        "evidence_recall_at_1",
                    ),
                    4,
                ),
                "evidence_recall_at_3": round(
                    mean_field(
                        format_records,
                        "evidence_recall_at_3",
                    ),
                    4,
                ),
                "evidence_recall_at_5": round(
                    mean_field(
                        format_records,
                        "evidence_recall_at_5",
                    ),
                    4,
                ),
                "evidence_recall_at_10": round(
                    mean_field(
                        format_records,
                        "evidence_recall_at_10",
                    ),
                    4,
                ),
                "evidence_mrr": round(
                    mean_field(
                        format_records,
                        "evidence_reciprocal_rank",
                    ),
                    4,
                ),

                "source_recall_at_1": round(
                    mean_field(
                        format_records,
                        "source_recall_at_1",
                    ),
                    4,
                ),
                "source_recall_at_5": round(
                    mean_field(
                        format_records,
                        "source_recall_at_5",
                    ),
                    4,
                ),
                "source_mrr": round(
                    mean_field(
                        format_records,
                        "source_reciprocal_rank",
                    ),
                    4,
                ),

                "mean_retrieval_time_ms": round(
                    statistics.fmean(
                        retrieval_times
                    ),
                    4,
                ),
                "p95_retrieval_time_ms": round(
                    percentile(
                        retrieval_times,
                        0.95,
                    ),
                    4,
                ),
            }
        )

    return format_summaries


# ============================================================
# Sauvegarde des résultats
# ============================================================

def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    Sauvegarde une liste de dictionnaires en CSV.
    """

    if not records:
        raise ValueError(
            f"Aucun résultat à écrire dans {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


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
# Benchmark pgvector
# ============================================================

def evaluate_question(
    retriever: PgVectorRetriever,
    row: dict[str, str],
    query_embedding: np.ndarray,
) -> dict[str, Any]:
    """
    Évalue une question avec pgvector.
    """

    start_time = time.perf_counter()

    results = retriever.search_by_embedding(
        query_embedding=query_embedding,
        top_k=TOP_K,
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
        "top_1_cosine_similarity": (
            round(
                top_result.cosine_similarity,
                6,
            )
            if top_result
            else ""
        ),
        "retrieval_time_ms": round(
            retrieval_time_ms,
            4,
        ),
    }


def main() -> None:
    """
    Lance le benchmark complet pgvector.
    """

    print("=" * 100)
    print(
        "BENCHMARK VECTOR RETRIEVAL — PGVECTOR"
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

    if len(questions) != len(
        query_embeddings
    ):
        raise ValueError(
            "Le nombre de questions ne correspond pas "
            "au nombre d'embeddings."
        )

    if len(questions) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"questions attendues, "
            f"{len(questions)} trouvées."
        )

    retriever = PgVectorRetriever(
        default_top_k=TOP_K
    )

    indexed_chunks = (
        retriever.count_indexed_chunks()
    )

    if indexed_chunks != 1478:
        raise ValueError(
            f"1478 chunks attendus dans PostgreSQL, "
            f"{indexed_chunks} trouvés."
        )

    print(
        "Méthode :",
        RETRIEVAL_METHOD,
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
        "Embeddings :",
        query_embeddings.shape,
    )

    print(
        "Top K :",
        TOP_K,
    )

    print()

    benchmark_start = time.perf_counter()

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
            f"{record['retrieval_time_ms']:.2f} ms"
        )

    total_benchmark_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    summary_record = build_summary(
        records=detail_records,
        indexed_chunks=indexed_chunks,
        total_benchmark_seconds=(
            total_benchmark_seconds
        ),
    )

    format_records = (
        build_format_summaries(
            detail_records
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
            "vector_retrieval_pgvector"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "retrieval_method": (
            RETRIEVAL_METHOD
        ),
        "database": "PostgreSQL + pgvector",
        "distance_operator": "<=>",
        "distance_metric": "cosine",
        "search_mode": "exact",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": (
            EXPECTED_EMBEDDING_DIMENSION
        ),
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
    print("BENCHMARK PGVECTOR TERMINÉ")
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