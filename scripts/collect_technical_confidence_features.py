from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.settings import get_settings
from src.confidence import (
    ConfidenceGate,
    ConfidenceThresholds,
)
from src.pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "08_confidence_gate_technical_ooc"
)

QUESTIONS_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_questions.csv"
)

FEATURES_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_features.csv"
)

SUMMARY_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_features_summary.json"
)

MANIFEST_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_features_manifest.json"
)

EXPECTED_NUMBER_OF_QUESTIONS = 60


OUTPUT_COLUMNS = [
    "id",
    "question",
    "expected_answerable",
    "category",
    "technology",
    "expected_source",
    "gold_evidence",
    "label_reason",

    "number_of_results",

    "top1_reranker_score",
    "top2_reranker_score",
    "reranker_margin",

    "top1_cosine_similarity",
    "top1_rrf_score",
    "mean_top_k_reranker_score",
    "supporting_results_count",

    "top1_source",
    "top1_chunk_id",
    "top1_hybrid_rank",
    "top1_vector_rank",
    "top1_bm25_rank",

    "embedding_time_ms",
    "hybrid_retrieval_time_ms",
    "reranking_time_ms",
    "total_time_ms",
]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def parse_boolean(
    value: str,
    field_name: str,
) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Valeur booléenne invalide pour "
        f"{field_name} : {value}"
    )


def load_questions(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier des questions ne contient "
                "pas d'en-tête."
            )

        required_columns = {
            "id",
            "question",
            "expected_answerable",
            "category",
            "technology",
            "expected_source",
            "gold_evidence",
            "label_reason",
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
            question_id = str(
                row.get("id", "")
            ).strip()

            question = str(
                row.get("question", "")
            ).strip()

            if not question_id:
                raise ValueError(
                    f"ID vide à la ligne {row_number}."
                )

            if not question:
                raise ValueError(
                    f"Question vide à la ligne "
                    f"{row_number}."
                )

            records.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_answerable": (
                        parse_boolean(
                            row.get(
                                "expected_answerable",
                                "",
                            ),
                            (
                                "expected_answerable "
                                f"ligne {row_number}"
                            ),
                        )
                    ),
                    "category": str(
                        row.get(
                            "category",
                            "",
                        )
                    ).strip(),
                    "technology": str(
                        row.get(
                            "technology",
                            "",
                        )
                    ).strip(),
                    "expected_source": str(
                        row.get(
                            "expected_source",
                            "",
                        )
                    ).strip(),
                    "gold_evidence": str(
                        row.get(
                            "gold_evidence",
                            "",
                        )
                    ).strip(),
                    "label_reason": str(
                        row.get(
                            "label_reason",
                            "",
                        )
                    ).strip(),
                }
            )

    if len(records) != (
        EXPECTED_NUMBER_OF_QUESTIONS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_QUESTIONS} "
            f"questions attendues, "
            f"{len(records)} trouvées."
        )

    return records


def load_existing_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return []

        return [
            {
                key: str(value or "")
                for key, value in row.items()
            }
            for row in reader
        ]


def write_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
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
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def optional_round(
    value: float | None,
    digits: int = 8,
) -> float | str:
    if value is None:
        return ""

    return round(
        float(value),
        digits,
    )


def calculate_summary(
    rows: list[dict[str, Any]],
    model_load_time_ms: float,
    warmup_time_ms: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(
            "Aucun résultat disponible."
        )

    answerable_rows = [
        row
        for row in rows
        if str(
            row["expected_answerable"]
        ).strip().lower() == "true"
    ]

    unanswerable_rows = [
        row
        for row in rows
        if str(
            row["expected_answerable"]
        ).strip().lower() == "false"
    ]

    total_times = [
        float(row["total_time_ms"])
        for row in rows
    ]

    answerable_top1 = [
        float(row["top1_reranker_score"])
        for row in answerable_rows
    ]

    unanswerable_top1 = [
        float(row["top1_reranker_score"])
        for row in unanswerable_rows
    ]

    answerable_rrf = [
        float(row["top1_rrf_score"])
        for row in answerable_rows
    ]

    unanswerable_rrf = [
        float(row["top1_rrf_score"])
        for row in unanswerable_rows
    ]

    answerable_margin = [
        float(row["reranker_margin"])
        for row in answerable_rows
    ]

    unanswerable_margin = [
        float(row["reranker_margin"])
        for row in unanswerable_rows
    ]

    return {
        "number_of_questions": len(rows),
        "number_of_answerable_questions": len(
            answerable_rows
        ),
        "number_of_unanswerable_questions": len(
            unanswerable_rows
        ),

        "model_load_time_ms": round(
            model_load_time_ms,
            4,
        ),
        "warmup_time_ms": round(
            warmup_time_ms,
            4,
        ),

        "mean_total_time_ms": round(
            statistics.mean(
                total_times
            ),
            4,
        ),
        "median_total_time_ms": round(
            statistics.median(
                total_times
            ),
            4,
        ),

        "answerable_mean_top1_score": round(
            statistics.mean(
                answerable_top1
            ),
            6,
        ),
        "answerable_median_top1_score": round(
            statistics.median(
                answerable_top1
            ),
            6,
        ),
        "unanswerable_mean_top1_score": round(
            statistics.mean(
                unanswerable_top1
            ),
            6,
        ),
        "unanswerable_median_top1_score": round(
            statistics.median(
                unanswerable_top1
            ),
            6,
        ),

        "answerable_mean_rrf_score": round(
            statistics.mean(
                answerable_rrf
            ),
            8,
        ),
        "unanswerable_mean_rrf_score": round(
            statistics.mean(
                unanswerable_rrf
            ),
            8,
        ),

        "answerable_mean_margin": round(
            statistics.mean(
                answerable_margin
            ),
            6,
        ),
        "unanswerable_mean_margin": round(
            statistics.mean(
                unanswerable_margin
            ),
            6,
        ),
    }


def main() -> None:
    print("=" * 100)
    print(
        "COLLECTE DES CARACTÉRISTIQUES — "
        "BENCHMARK 2 TECHNIQUE HORS CORPUS"
    )
    print("=" * 100)

    require_file(
        QUESTIONS_PATH
    )

    BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_questions(
        QUESTIONS_PATH
    )

    existing_rows = load_existing_rows(
        FEATURES_PATH
    )

    completed_ids = {
        row["id"]
        for row in existing_rows
    }

    print(
        "Questions totales :",
        len(questions),
    )
    print(
        "Questions déjà traitées :",
        len(completed_ids),
    )
    print(
        "Questions restantes :",
        len(questions)
        - len(completed_ids),
    )
    print()

    settings = get_settings(
        force_reload=True
    )

    pipeline = (
        RetrievalPipeline.from_settings(
            settings
        )
    )

    feature_extractor = ConfidenceGate(
        thresholds=ConfidenceThresholds(
            minimum_top1_score=0.0,
            minimum_margin=0.0,
            minimum_cosine_similarity=None,
            minimum_rrf_score=None,
            minimum_supporting_results=0,
        )
    )

    print(
        "Embedding :",
        pipeline.embedding_model_name,
    )
    print(
        "Reranker :",
        pipeline.reranker_model_name,
    )
    print(
        "Device embedding :",
        pipeline.embedding_device,
    )
    print(
        "Device reranker :",
        pipeline.reranker_device,
    )
    print()

    model_load_start = (
        time.perf_counter()
    )

    pipeline.load_embedding_model()
    pipeline.reranker.load_model()

    model_load_time_ms = (
        time.perf_counter()
        - model_load_start
    ) * 1000

    print(
        "Chargement des modèles :",
        round(
            model_load_time_ms,
            2,
        ),
        "ms",
    )

    warmup_start = (
        time.perf_counter()
    )

    pipeline.retrieve(
        (
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        )
    )

    warmup_time_ms = (
        time.perf_counter()
        - warmup_start
    ) * 1000

    print(
        "Warm-up :",
        round(
            warmup_time_ms,
            2,
        ),
        "ms",
    )
    print()

    output_rows: list[dict[str, Any]] = [
        dict(row)
        for row in existing_rows
    ]

    remaining_questions = [
        record
        for record in questions
        if record["id"]
        not in completed_ids
    ]

    total_remaining = len(
        remaining_questions
    )

    for index, record in enumerate(
        remaining_questions,
        start=1,
    ):
        question_id = record["id"]
        question = record["question"]

        try:
            response = pipeline.retrieve(
                question
            )

            features = (
                feature_extractor
                .extract_features(
                    response.results
                )
            )

            top1 = (
                response.results[0]
                if response.results
                else None
            )

            row = {
                "id": question_id,
                "question": question,
                "expected_answerable": (
                    "true"
                    if record[
                        "expected_answerable"
                    ]
                    else "false"
                ),
                "category": (
                    record["category"]
                ),
                "technology": (
                    record["technology"]
                ),
                "expected_source": (
                    record["expected_source"]
                ),
                "gold_evidence": (
                    record["gold_evidence"]
                ),
                "label_reason": (
                    record["label_reason"]
                ),

                "number_of_results": (
                    features.number_of_results
                ),

                "top1_reranker_score": round(
                    features
                    .top1_reranker_score,
                    8,
                ),
                "top2_reranker_score": (
                    optional_round(
                        features
                        .top2_reranker_score
                    )
                ),
                "reranker_margin": round(
                    features.reranker_margin,
                    8,
                ),

                "top1_cosine_similarity": (
                    optional_round(
                        features
                        .top1_cosine_similarity
                    )
                ),
                "top1_rrf_score": (
                    optional_round(
                        features
                        .top1_rrf_score,
                        10,
                    )
                ),
                "mean_top_k_reranker_score": (
                    round(
                        features
                        .mean_top_k_reranker_score,
                        8,
                    )
                ),
                "supporting_results_count": (
                    features
                    .supporting_results_count
                ),

                "top1_source": (
                    top1.source
                    if top1
                    else ""
                ),
                "top1_chunk_id": (
                    top1.chunk_id
                    if top1
                    else ""
                ),
                "top1_hybrid_rank": (
                    top1.hybrid_rank
                    if top1
                    else ""
                ),
                "top1_vector_rank": (
                    top1.vector_rank
                    if (
                        top1
                        and top1.vector_rank
                        is not None
                    )
                    else ""
                ),
                "top1_bm25_rank": (
                    top1.bm25_rank
                    if (
                        top1
                        and top1.bm25_rank
                        is not None
                    )
                    else ""
                ),

                "embedding_time_ms": (
                    response.timings
                    .embedding_time_ms
                ),
                "hybrid_retrieval_time_ms": (
                    response.timings
                    .hybrid_retrieval_time_ms
                ),
                "reranking_time_ms": (
                    response.timings
                    .reranking_time_ms
                ),
                "total_time_ms": (
                    response.timings
                    .total_time_ms
                ),
            }

            output_rows.append(row)

            write_rows(
                FEATURES_PATH,
                output_rows,
            )

            label = (
                "answerable"
                if record[
                    "expected_answerable"
                ]
                else "technical_ooc"
            )

            print(
                f"[{index:02d}/"
                f"{total_remaining:02d}] "
                f"{question_id} | "
                f"{label} | "
                f"Top1: "
                f"{features.top1_reranker_score:.4f} | "
                f"RRF: "
                f"{features.top1_rrf_score} | "
                f"Margin: "
                f"{features.reranker_margin:.4f} | "
                f"Total: "
                f"{response.timings.total_time_ms:.2f} ms"
            )

        except Exception as error:
            write_rows(
                FEATURES_PATH,
                output_rows,
            )

            raise RuntimeError(
                "Erreur pendant le traitement de "
                f"{question_id}. "
                "Les résultats précédents restent "
                f"sauvegardés dans {FEATURES_PATH}."
            ) from error

    summary = calculate_summary(
        rows=output_rows,
        model_load_time_ms=(
            model_load_time_ms
        ),
        warmup_time_ms=(
            warmup_time_ms
        ),
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    manifest = {
        "benchmark_name": (
            "confidence_gate_technical_ooc_feature_collection"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "company": (
            settings.corpus.company
        ),
        "settings_path": str(
            settings.source_path
        ),
        "number_of_questions": len(
            output_rows
        ),
        "embedding_model": (
            pipeline.embedding_model_name
        ),
        "reranker_model": (
            pipeline.reranker_model_name
        ),
        "candidate_k": (
            pipeline.candidate_k
        ),
        "hybrid_top_k": (
            pipeline.hybrid_top_k
        ),
        "final_top_k": (
            pipeline.final_top_k
        ),
        "files": {
            "questions": str(
                QUESTIONS_PATH
            ),
            "features": str(
                FEATURES_PATH
            ),
            "summary": str(
                SUMMARY_PATH
            ),
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("COLLECTE BENCHMARK 2 TERMINÉE")
    print("=" * 100)

    for key, value in summary.items():
        print(
            f"{key} : {value}"
        )

    print()
    print(
        "Caractéristiques :",
        FEATURES_PATH,
    )
    print(
        "Résumé :",
        SUMMARY_PATH,
    )
    print(
        "Manifeste :",
        MANIFEST_PATH,
    )

    pipeline.unload_models()


if __name__ == "__main__":
    main()