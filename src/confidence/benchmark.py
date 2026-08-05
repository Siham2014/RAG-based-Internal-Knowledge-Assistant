from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.confidence.metrics import (
    ConfidenceMetrics,
    calculate_confidence_metrics,
)


@dataclass(frozen=True)
class ConfidenceFeatureRecord:
    """
    Une question accompagnée des caractéristiques calculées
    par le pipeline documentaire.
    """

    question_id: str
    question: str
    expected_answerable: bool
    category: str

    top1_reranker_score: float
    top2_reranker_score: float | None
    reranker_margin: float

    top1_cosine_similarity: float | None
    top1_rrf_score: float | None

    mean_top_k_reranker_score: float
    supporting_results_count: int


@dataclass(frozen=True)
class ConfidenceBenchmarkConfiguration:
    """
    Une combinaison de seuils testée par le benchmark.
    """

    minimum_top1_score: float
    minimum_margin: float
    minimum_cosine_similarity: float | None
    minimum_rrf_score: float | None
    minimum_supporting_results: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_top1_score": self.minimum_top1_score,
            "minimum_margin": self.minimum_margin,
            "minimum_cosine_similarity": (
                self.minimum_cosine_similarity
            ),
            "minimum_rrf_score": self.minimum_rrf_score,
            "minimum_supporting_results": (
                self.minimum_supporting_results
            ),
        }


@dataclass(frozen=True)
class ConfidenceBenchmarkResult:
    """
    Résultat d'une configuration de seuils.
    """

    configuration: ConfidenceBenchmarkConfiguration
    metrics: ConfidenceMetrics
    balanced_error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.configuration.to_dict(),
            **self.metrics.to_dict(),
            "balanced_error_rate": self.balanced_error_rate,
        }


def parse_boolean(
    value: str,
    field_name: str,
) -> bool:
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Valeur booléenne invalide pour {field_name}: {value}"
    )


def parse_optional_float(
    value: str | None,
) -> float | None:
    if value is None:
        return None

    normalized = str(value).strip()

    if not normalized:
        return None

    return float(normalized)


def load_feature_records(
    path: Path,
) -> list[ConfidenceFeatureRecord]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier de caractéristiques introuvable : {path}"
        )

    records: list[ConfidenceFeatureRecord] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier de caractéristiques ne contient pas d'en-tête."
            )

        required_columns = {
            "id",
            "question",
            "expected_answerable",
            "category",
            "top1_reranker_score",
            "top2_reranker_score",
            "reranker_margin",
            "top1_cosine_similarity",
            "top1_rrf_score",
            "mean_top_k_reranker_score",
            "supporting_results_count",
        }

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Colonnes manquantes : "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            records.append(
                ConfidenceFeatureRecord(
                    question_id=str(row["id"]).strip(),
                    question=str(row["question"]).strip(),
                    expected_answerable=parse_boolean(
                        row["expected_answerable"],
                        f"expected_answerable ligne {row_number}",
                    ),
                    category=str(row["category"]).strip(),
                    top1_reranker_score=float(
                        row["top1_reranker_score"]
                    ),
                    top2_reranker_score=parse_optional_float(
                        row["top2_reranker_score"]
                    ),
                    reranker_margin=float(
                        row["reranker_margin"]
                    ),
                    top1_cosine_similarity=parse_optional_float(
                        row["top1_cosine_similarity"]
                    ),
                    top1_rrf_score=parse_optional_float(
                        row["top1_rrf_score"]
                    ),
                    mean_top_k_reranker_score=float(
                        row["mean_top_k_reranker_score"]
                    ),
                    supporting_results_count=int(
                        row["supporting_results_count"]
                    ),
                )
            )

    if not records:
        raise ValueError(
            "Aucun enregistrement trouvé dans le fichier."
        )

    return records


def predict_answerable(
    record: ConfidenceFeatureRecord,
    configuration: ConfidenceBenchmarkConfiguration,
) -> bool:
    if (
        record.top1_reranker_score
        < configuration.minimum_top1_score
    ):
        return False

    if (
        record.reranker_margin
        < configuration.minimum_margin
    ):
        return False

    if (
        configuration.minimum_cosine_similarity
        is not None
    ):
        cosine = (
            record.top1_cosine_similarity
            if record.top1_cosine_similarity is not None
            else 0.0
        )

        if cosine < configuration.minimum_cosine_similarity:
            return False

    if configuration.minimum_rrf_score is not None:
        rrf_score = (
            record.top1_rrf_score
            if record.top1_rrf_score is not None
            else 0.0
        )

        if rrf_score < configuration.minimum_rrf_score:
            return False

    if (
        record.supporting_results_count
        < configuration.minimum_supporting_results
    ):
        return False

    return True


def generate_configurations(
    top1_thresholds: Iterable[float],
    margin_thresholds: Iterable[float],
    cosine_thresholds: Iterable[float | None],
    rrf_thresholds: Iterable[float | None],
    supporting_thresholds: Iterable[int],
) -> list[ConfidenceBenchmarkConfiguration]:
    configurations: list[
        ConfidenceBenchmarkConfiguration
    ] = []

    for top1 in top1_thresholds:
        for margin in margin_thresholds:
            for cosine in cosine_thresholds:
                for rrf in rrf_thresholds:
                    for supporting in supporting_thresholds:
                        configurations.append(
                            ConfidenceBenchmarkConfiguration(
                                minimum_top1_score=float(top1),
                                minimum_margin=float(margin),
                                minimum_cosine_similarity=(
                                    None
                                    if cosine is None
                                    else float(cosine)
                                ),
                                minimum_rrf_score=(
                                    None
                                    if rrf is None
                                    else float(rrf)
                                ),
                                minimum_supporting_results=int(
                                    supporting
                                ),
                            )
                        )

    return configurations


def evaluate_configuration(
    records: list[ConfidenceFeatureRecord],
    configuration: ConfidenceBenchmarkConfiguration,
) -> ConfidenceBenchmarkResult:
    expected_labels = [
        record.expected_answerable
        for record in records
    ]

    predicted_labels = [
        predict_answerable(
            record=record,
            configuration=configuration,
        )
        for record in records
    ]

    metrics = calculate_confidence_metrics(
        expected_labels=expected_labels,
        predicted_labels=predicted_labels,
    )

    balanced_error_rate = round(
        (
            metrics.false_acceptance_rate
            + metrics.false_rejection_rate
        )
        / 2.0,
        6,
    )

    return ConfidenceBenchmarkResult(
        configuration=configuration,
        metrics=metrics,
        balanced_error_rate=balanced_error_rate,
    )


def run_confidence_benchmark(
    records: list[ConfidenceFeatureRecord],
    configurations: list[
        ConfidenceBenchmarkConfiguration
    ],
) -> list[ConfidenceBenchmarkResult]:
    if not configurations:
        raise ValueError(
            "Aucune configuration fournie au benchmark."
        )

    return [
        evaluate_configuration(
            records=records,
            configuration=configuration,
        )
        for configuration in configurations
    ]


def rank_results(
    results: list[ConfidenceBenchmarkResult],
) -> list[ConfidenceBenchmarkResult]:
    """
    Critères de classement :

    1. plus faible taux de fausse acceptation ;
    2. meilleur F1-score ;
    3. meilleure accuracy ;
    4. plus faible balanced error rate ;
    5. meilleur recall ;
    6. plus grande couverture.
    """

    return sorted(
        results,
        key=lambda result: (
            result.metrics.false_acceptance_rate,
            -result.metrics.f1_score,
            -result.metrics.accuracy,
            result.balanced_error_rate,
            -result.metrics.recall,
            -result.metrics.coverage,
            result.configuration.minimum_top1_score,
            result.configuration.minimum_margin,
        ),
    )


def write_benchmark_results(
    path: Path,
    results: list[ConfidenceBenchmarkResult],
) -> None:
    rows = [
        result.to_dict()
        for result in results
    ]

    if not rows:
        raise ValueError(
            "Aucun résultat à sauvegarder."
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_best_result(
    path: Path,
    result: ConfidenceBenchmarkResult,
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
            result.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )