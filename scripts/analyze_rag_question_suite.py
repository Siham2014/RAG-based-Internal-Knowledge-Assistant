from __future__ import annotations

import csv
from pathlib import Path

from src.common.settings import PROJECT_ROOT


DETAILS_PATH = (
    PROJECT_ROOT
    / "benchmarking"
    / "09_rag_question_suite"
    / "rag_question_suite_details.csv"
)


def normalize_boolean(
    value: str,
) -> bool | None:
    normalized = str(
        value or ""
    ).strip().lower()

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

    return None


def print_section(
    title: str,
) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def print_row(
    row: dict[str, str],
) -> None:
    print()
    print("Question ID :", row["question_id"])
    print("Catégorie :", row["category"])
    print("Question :", row["question"])
    print(
        "Expected answerable :",
        row["expected_answerable"],
    )
    print(
        "Predicted answerable :",
        row["predicted_answerable"],
    )
    print(
        "Accepted :",
        row["accepted"],
    )
    print(
        "Confidence :",
        row["confidence_score"],
    )
    print(
        "Top1 reranker :",
        row["top1_reranker_score"],
    )
    print(
        "Margin :",
        row["reranker_margin"],
    )
    print(
        "RRF :",
        row["top1_rrf_score"],
    )
    print(
        "Failed rules :",
        row["failed_rules"] or "Aucune",
    )
    print(
        "Provider :",
        row["provider"] or "Aucun",
    )
    print(
        "Citations valid :",
        row["citations_valid"],
    )
    print(
        "Citations :",
        row["citations"] or "Aucune",
    )
    print(
        "Top source :",
        row["top_source"] or "Aucune",
    )
    print(
        "Top chunk :",
        row["top_chunk_id"] or "Aucun",
    )
    print(
        "Refusal reason :",
        row["refusal_reason"] or "Aucune",
    )
    print("Réponse :")
    print(row["answer"] or "Aucune réponse")


def main() -> None:
    if not DETAILS_PATH.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {DETAILS_PATH}"
        )

    with DETAILS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    classification_errors = [
        row
        for row in rows
        if not normalize_boolean(
            row["prediction_correct"]
        )
    ]

    invalid_citations = [
        row
        for row in rows
        if (
            normalize_boolean(
                row["llm_called"]
            )
            and normalize_boolean(
                row["citations_valid"]
            )
            is False
        )
    ]

    llm_failures = [
        row
        for row in rows
        if (
            normalize_boolean(
                row["llm_called"]
            )
            and not normalize_boolean(
                row["llm_success"]
            )
        )
    ]

    print_section(
        "ERREURS DE CLASSIFICATION"
    )

    if not classification_errors:
        print("Aucune erreur de classification.")
    else:
        for row in classification_errors:
            print_row(row)

    print_section(
        "RÉPONSES AVEC CITATIONS INVALIDES"
    )

    if not invalid_citations:
        print("Aucune citation invalide.")
    else:
        for row in invalid_citations:
            print_row(row)

    print_section(
        "ÉCHECS TECHNIQUES DU LLM"
    )

    if not llm_failures:
        print("Aucun échec technique du LLM.")
    else:
        for row in llm_failures:
            print_row(row)

    print()
    print("=" * 110)
    print("RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 110)
    print(
        "Erreurs de classification :",
        len(classification_errors),
    )
    print(
        "Citations invalides :",
        len(invalid_citations),
    )
    print(
        "Échecs techniques LLM :",
        len(llm_failures),
    )


if __name__ == "__main__":
    main()