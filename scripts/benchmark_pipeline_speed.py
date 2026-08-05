from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.common.settings import get_settings
from src.pipeline import RetrievalPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_30_questions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "06_pipeline_speed"
)

DETAILS_PATH = (
    OUTPUT_DIR
    / "pipeline_speed_details.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "pipeline_speed_summary.json"
)

# Pour un premier test local sur CPU.
NUMBER_OF_QUESTIONS = 10

# Une question de chauffe, non comptée dans les métriques.
WARMUP_QUESTION = (
    "Who shares responsibility for the "
    "sustainability of cloud workloads?"
)


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
                "Le fichier d'évaluation ne contient "
                "pas d'en-tête."
            )

        for row in reader:
            question_id = str(
                row.get("id", "")
            ).strip()

            question = str(
                row.get("question", "")
            ).strip()

            if question_id and question:
                questions.append(
                    {
                        "id": question_id,
                        "question": question,
                        "expected_source": str(
                            row.get(
                                "expected_source",
                                "",
                            )
                        ).strip(),
                    }
                )

    if not questions:
        raise ValueError(
            "Aucune question valide trouvée."
        )

    return questions


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * percentile_value

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

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
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 100)
    print("BENCHMARK DE VITESSE DU RETRIEVAL PIPELINE")
    print("=" * 100)

    require_file(EVALUATION_PATH)

    settings = get_settings(
        force_reload=True
    )

    questions = load_questions(
        EVALUATION_PATH
    )

    selected_questions = questions[
        :NUMBER_OF_QUESTIONS
    ]

    pipeline = RetrievalPipeline.from_settings(
        settings
    )

    print("Configuration :", settings.source_path)
    print("Entreprise :", settings.corpus.company)
    print("Embedding :", pipeline.embedding_model_name)
    print("Reranker :", pipeline.reranker_model_name)
    print("Embedding device :", pipeline.embedding_device)
    print("Reranker device :", pipeline.reranker_device)
    print("Questions mesurées :", len(selected_questions))
    print()

    # ========================================================
    # Chargement et warm-up
    # ========================================================

    print("=" * 100)
    print("CHARGEMENT ET WARM-UP")
    print("=" * 100)

    startup_start = time.perf_counter()

    pipeline.load_embedding_model()
    pipeline.reranker.load_model()

    model_load_time_ms = (
        time.perf_counter()
        - startup_start
    ) * 1000

    print(
        "Temps de chargement des modèles :",
        round(model_load_time_ms, 2),
        "ms",
    )

    warmup_start = time.perf_counter()

    warmup_response = pipeline.retrieve(
        WARMUP_QUESTION
    )

    warmup_time_ms = (
        time.perf_counter()
        - warmup_start
    ) * 1000

    print(
        "Temps du warm-up :",
        round(warmup_time_ms, 2),
        "ms",
    )

    print(
        "Résultats warm-up :",
        len(warmup_response.results),
    )

    print()

    # ========================================================
    # Mesures réelles
    # ========================================================

    detail_rows: list[
        dict[str, Any]
    ] = []

    for index, record in enumerate(
        selected_questions,
        start=1,
    ):
        question_id = record["id"]
        question = record["question"]

        response = pipeline.retrieve(
            question
        )

        top_result = (
            response.results[0]
            if response.results
            else None
        )

        row = {
            "id": question_id,
            "question": question,
            "number_of_results": len(
                response.results
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
            "top1_source": (
                top_result.source
                if top_result
                else ""
            ),
            "top1_reranker_score": (
                round(
                    top_result.reranker_score,
                    6,
                )
                if top_result
                else ""
            ),
            "top1_rrf_score": (
                round(
                    top_result.rrf_score,
                    8,
                )
                if top_result
                else ""
            ),
            "top1_cosine_similarity": (
                round(
                    top_result.cosine_similarity,
                    6,
                )
                if (
                    top_result
                    and top_result
                    .cosine_similarity
                    is not None
                )
                else ""
            ),
        }

        detail_rows.append(row)

        print(
            f"[{index:02d}/"
            f"{len(selected_questions):02d}] "
            f"{question_id} | "
            f"Embedding: "
            f"{row['embedding_time_ms']:.2f} ms | "
            f"Retrieval: "
            f"{row['hybrid_retrieval_time_ms']:.2f} ms | "
            f"Reranking: "
            f"{row['reranking_time_ms']:.2f} ms | "
            f"Total: "
            f"{row['total_time_ms']:.2f} ms"
        )

    embedding_times = [
        float(row["embedding_time_ms"])
        for row in detail_rows
    ]

    retrieval_times = [
        float(
            row[
                "hybrid_retrieval_time_ms"
            ]
        )
        for row in detail_rows
    ]

    reranking_times = [
        float(row["reranking_time_ms"])
        for row in detail_rows
    ]

    total_times = [
        float(row["total_time_ms"])
        for row in detail_rows
    ]

    summary = {
        "company": settings.corpus.company,
        "settings_path": str(
            settings.source_path
        ),
        "number_of_questions": len(
            detail_rows
        ),
        "embedding_model": (
            pipeline.embedding_model_name
        ),
        "reranker_model": (
            pipeline.reranker_model_name
        ),
        "embedding_device": (
            pipeline.embedding_device
        ),
        "reranker_device": (
            pipeline.reranker_device
        ),
        "model_load_time_ms": round(
            model_load_time_ms,
            4,
        ),
        "warmup_time_ms": round(
            warmup_time_ms,
            4,
        ),
        "mean_embedding_time_ms": round(
            statistics.mean(
                embedding_times
            ),
            4,
        ),
        "mean_hybrid_retrieval_time_ms": round(
            statistics.mean(
                retrieval_times
            ),
            4,
        ),
        "mean_reranking_time_ms": round(
            statistics.mean(
                reranking_times
            ),
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
        "p95_total_time_ms": round(
            percentile(
                total_times,
                0.95,
            ),
            4,
        ),
        "minimum_total_time_ms": round(
            min(total_times),
            4,
        ),
        "maximum_total_time_ms": round(
            max(total_times),
            4,
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        DETAILS_PATH,
        detail_rows,
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

    print()
    print("=" * 100)
    print("RÉSUMÉ")
    print("=" * 100)

    for key, value in summary.items():
        print(f"{key} : {value}")

    print()
    print("Détails :", DETAILS_PATH)
    print("Résumé :", SUMMARY_PATH)

    pipeline.unload_models()


if __name__ == "__main__":
    main()