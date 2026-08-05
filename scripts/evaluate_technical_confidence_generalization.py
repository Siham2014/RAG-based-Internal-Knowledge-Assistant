from __future__ import annotations

import csv
import json
from pathlib import Path

from src.confidence import (
    ConfidenceBenchmarkConfiguration,
    evaluate_configuration,
    load_feature_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "08_confidence_gate_technical_ooc"
)

FEATURES_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_features.csv"
)

OUTPUT_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_generalization.json"
)

DETAILS_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_technical_predictions.csv"
)


REFERENCE_CONFIGURATION = (
    ConfidenceBenchmarkConfiguration(
        minimum_top1_score=0.30,
        minimum_margin=0.00,
        minimum_cosine_similarity=None,
        minimum_rrf_score=0.015,
        minimum_supporting_results=0,
    )
)


def predict_record(record) -> bool:
    if (
        record.top1_reranker_score
        < REFERENCE_CONFIGURATION.minimum_top1_score
    ):
        return False

    if (
        record.reranker_margin
        < REFERENCE_CONFIGURATION.minimum_margin
    ):
        return False

    if (
        REFERENCE_CONFIGURATION.minimum_rrf_score
        is not None
    ):
        rrf_score = (
            record.top1_rrf_score
            if record.top1_rrf_score is not None
            else 0.0
        )

        if (
            rrf_score
            < REFERENCE_CONFIGURATION.minimum_rrf_score
        ):
            return False

    return True


def main() -> None:
    print("=" * 100)
    print(
        "ÉVALUATION DE GÉNÉRALISATION "
        "DU CONFIDENCE GATE"
    )
    print("=" * 100)

    records = load_feature_records(
        FEATURES_PATH
    )

    result = evaluate_configuration(
        records=records,
        configuration=REFERENCE_CONFIGURATION,
    )

    prediction_rows = []

    for record in records:
        predicted = predict_record(
            record
        )

        prediction_rows.append(
            {
                "id": record.question_id,
                "question": record.question,
                "category": record.category,
                "expected_answerable": (
                    record.expected_answerable
                ),
                "predicted_answerable": predicted,
                "correct": (
                    predicted
                    == record.expected_answerable
                ),
                "top1_reranker_score": (
                    record.top1_reranker_score
                ),
                "reranker_margin": (
                    record.reranker_margin
                ),
                "top1_cosine_similarity": (
                    record.top1_cosine_similarity
                ),
                "top1_rrf_score": (
                    record.top1_rrf_score
                ),
                "supporting_results_count": (
                    record.supporting_results_count
                ),
            }
        )

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                prediction_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            prediction_rows
        )

    output = {
        "benchmark": (
            "technical_out_of_corpus_generalization"
        ),
        "number_of_questions": len(records),
        "reference_configuration": (
            REFERENCE_CONFIGURATION.to_dict()
        ),
        "metrics": result.metrics.to_dict(),
        "balanced_error_rate": (
            result.balanced_error_rate
        ),
        "files": {
            "features": str(FEATURES_PATH),
            "predictions": str(DETAILS_PATH),
        },
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("CONFIGURATION APPLIQUÉE")
    print("-" * 100)

    for key, value in (
        REFERENCE_CONFIGURATION
        .to_dict()
        .items()
    ):
        print(f"{key} : {value}")

    print()
    print("RÉSULTATS")
    print("-" * 100)

    for key, value in (
        result.metrics
        .to_dict()
        .items()
    ):
        print(f"{key} : {value}")

    print(
        "balanced_error_rate :",
        result.balanced_error_rate,
    )

    print()
    print("Prédictions :", DETAILS_PATH)
    print("Résumé :", OUTPUT_PATH)


if __name__ == "__main__":
    main()