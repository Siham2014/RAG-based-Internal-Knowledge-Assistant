from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.confidence import (
    generate_configurations,
    load_feature_records,
    rank_results,
    run_confidence_benchmark,
    write_benchmark_results,
    write_best_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "07_confidence_gate_benchmark"
)

FEATURES_PATH = (
    BENCHMARK_DIR
    / "confidence_gate_features.csv"
)

RESULTS_PATH = (
    BENCHMARK_DIR
    / "confidence_thresholds_results.csv"
)

BEST_PATH = (
    BENCHMARK_DIR
    / "confidence_thresholds_best.json"
)

TOP_RESULTS_PATH = (
    BENCHMARK_DIR
    / "confidence_thresholds_top20.csv"
)

MANIFEST_PATH = (
    BENCHMARK_DIR
    / "confidence_thresholds_manifest.json"
)


TOP1_THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

MARGIN_THRESHOLDS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

COSINE_THRESHOLDS = [
    None,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]

RRF_THRESHOLDS = [
    None,
    0.015,
    0.020,
    0.025,
    0.030,
]

SUPPORTING_THRESHOLDS = [
    0,
    1,
    2,
    3,
]


def main() -> None:
    print("=" * 100)
    print("BENCHMARK DES SEUILS DU CONFIDENCE GATE")
    print("=" * 100)

    records = load_feature_records(
        FEATURES_PATH
    )

    configurations = generate_configurations(
        top1_thresholds=TOP1_THRESHOLDS,
        margin_thresholds=MARGIN_THRESHOLDS,
        cosine_thresholds=COSINE_THRESHOLDS,
        rrf_thresholds=RRF_THRESHOLDS,
        supporting_thresholds=SUPPORTING_THRESHOLDS,
    )

    print("Questions :", len(records))
    print(
        "Configurations testées :",
        len(configurations),
    )

    results = run_confidence_benchmark(
        records=records,
        configurations=configurations,
    )

    ranked_results = rank_results(
        results
    )

    best_result = ranked_results[0]

    write_benchmark_results(
        RESULTS_PATH,
        ranked_results,
    )

    write_benchmark_results(
        TOP_RESULTS_PATH,
        ranked_results[:20],
    )

    write_best_result(
        BEST_PATH,
        best_result,
    )

    manifest = {
        "benchmark_name": (
            "confidence_gate_threshold_search"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "number_of_questions": len(records),
        "number_of_configurations": len(
            configurations
        ),
        "selection_priority": [
            "minimum_false_acceptance_rate",
            "maximum_f1_score",
            "maximum_accuracy",
            "minimum_balanced_error_rate",
            "maximum_recall",
            "maximum_coverage",
        ],
        "threshold_grids": {
            "top1": TOP1_THRESHOLDS,
            "margin": MARGIN_THRESHOLDS,
            "cosine": COSINE_THRESHOLDS,
            "rrf": RRF_THRESHOLDS,
            "supporting_results": (
                SUPPORTING_THRESHOLDS
            ),
        },
        "files": {
            "features": str(FEATURES_PATH),
            "results": str(RESULTS_PATH),
            "top20": str(TOP_RESULTS_PATH),
            "best": str(BEST_PATH),
        },
        "best_result": best_result.to_dict(),
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
    print("MEILLEURE CONFIGURATION")
    print("=" * 100)

    for key, value in best_result.to_dict().items():
        print(f"{key} : {value}")

    print()
    print("Résultats complets :", RESULTS_PATH)
    print("Top 20 :", TOP_RESULTS_PATH)
    print("Meilleur résultat :", BEST_PATH)
    print("Manifeste :", MANIFEST_PATH)


if __name__ == "__main__":
    main()