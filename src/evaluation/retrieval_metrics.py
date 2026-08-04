from __future__ import annotations

import csv
import math
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Protocol


class RetrievalResult(Protocol):
    """
    Structure minimale attendue pour un résultat de retrieval.

    PgVectorSearchResult, BM25SearchResult et les futurs
    résultats Hybrid peuvent tous respecter cette interface.
    """

    rank: int
    chunk_id: str
    content: str
    source: str
    document_format: str


def normalize_text(text: str) -> str:
    """
    Normalise un texte avant la comparaison d'une preuve.

    La normalisation ignore :
    - les majuscules/minuscules ;
    - les espaces multiples ;
    - les retours à la ligne ;
    - certaines variantes Unicode.
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
    Retourne uniquement le nom du fichier source.

    Cela permet de comparer :
        markdown\\document.md
    avec :
        document.md
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

    source = row.get(
        "expected_source",
        "",
    ).lower()

    if source.endswith(".md"):
        return "markdown"

    if source.endswith(".html"):
        return "html"

    if source.endswith(".pdf"):
        return "pdf"

    return "unknown"


def find_source_rank(
    results: list[RetrievalResult],
    expected_source: str,
) -> int | None:
    """
    Retourne le premier rang appartenant à la source attendue.
    """

    expected_filename = canonical_filename(
        expected_source
    )

    for result in results:

        if (
            canonical_filename(result.source)
            == expected_filename
        ):
            return int(result.rank)

    return None


def find_evidence_rank(
    results: list[RetrievalResult],
    gold_evidence: str,
) -> int | None:
    """
    Retourne le premier rang dont le contenu contient
    la preuve de référence.
    """

    normalized_evidence = normalize_text(
        gold_evidence
    )

    for result in results:

        normalized_content = normalize_text(
            result.content
        )

        if normalized_evidence in normalized_content:
            return int(result.rank)

    return None


def recall_at_k(
    rank: int | None,
    k: int,
) -> int:
    """
    Retourne 1 lorsque le résultat pertinent apparaît
    dans les k premiers résultats.
    """

    if rank is None:
        return 0

    return int(rank <= k)


def reciprocal_rank(
    rank: int | None,
) -> float:
    """
    Calcule l'inverse du rang.
    """

    if rank is None:
        return 0.0

    return 1.0 / rank


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    """
    Calcule un percentile avec interpolation linéaire.
    """

    if not values:
        return 0.0

    sorted_values = sorted(
        float(value)
        for value in values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        len(sorted_values) - 1
    ) * percentile_value

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[
        lower_index
    ]

    upper_value = sorted_values[
        upper_index
    ]

    fraction = position - lower_index

    return (
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
    retrieval_method: str,
    number_of_indexed_chunks: int,
    total_benchmark_seconds: float,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construit le résumé global d'un benchmark de retrieval.
    """

    if not records:
        raise ValueError(
            "Aucun résultat disponible pour construire le résumé."
        )

    retrieval_times = [
        float(record["retrieval_time_ms"])
        for record in records
    ]

    summary: dict[str, Any] = {
        "retrieval_method": retrieval_method,
        "number_of_indexed_chunks": (
            number_of_indexed_chunks
        ),
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

    if extra_fields:
        summary.update(
            extra_fields
        )

    return summary


def build_format_summaries(
    records: list[dict[str, Any]],
    retrieval_method: str,
) -> list[dict[str, Any]]:
    """
    Calcule les métriques séparément pour HTML,
    Markdown et PDF.
    """

    formats = sorted(
        {
            str(record["format"])
            for record in records
        }
    )

    summaries: list[dict[str, Any]] = []

    for document_format in formats:

        format_records = [
            record
            for record in records
            if record["format"]
            == document_format
        ]

        retrieval_times = [
            float(record["retrieval_time_ms"])
            for record in format_records
        ]

        summaries.append(
            {
                "retrieval_method": retrieval_method,
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
                "source_recall_at_3": round(
                    mean_field(
                        format_records,
                        "source_recall_at_3",
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
                "source_recall_at_10": round(
                    mean_field(
                        format_records,
                        "source_recall_at_10",
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

    return summaries


def load_evaluation_questions(
    path: Path,
) -> list[dict[str, str]]:
    """
    Charge et valide le fichier CSV des questions.
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
                "Le CSV ne contient pas d'en-tête."
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

                if not normalized_row.get(
                    required_column
                ):
                    raise ValueError(
                        f"Valeur vide pour "
                        f"'{required_column}' "
                        f"à la ligne {row_number}."
                    )

            questions.append(
                normalized_row
            )

    return questions


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

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(records)