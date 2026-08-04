from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "03_retrieval_benchmark"
)

INPUT_FILES = {
    "pgvector_exact": RESULTS_DIR / "pgvector_summary.csv",
    "sqlite_fts5_bm25": RESULTS_DIR / "bm25_summary.csv",
    "hybrid_rrf": RESULTS_DIR / "hybrid_summary.csv",
}

OUTPUT_PATH = (
    RESULTS_DIR
    / "retrieval_benchmark_comparison.csv"
)


def read_first_csv_row(
    path: Path,
) -> dict[str, str]:
    """
    Lit la première ligne de données d'un CSV.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        row = next(reader, None)

    if row is None:
        raise ValueError(
            f"Le fichier ne contient aucune donnée : {path}"
        )

    return {
        key: str(value or "").strip()
        for key, value in row.items()
        if key is not None
    }


def to_float(
    value: str,
) -> float:
    """
    Convertit une valeur CSV en nombre.
    """

    return float(
        str(value).replace(",", ".")
    )


def main() -> None:
    """
    Construit le tableau comparatif final.
    """

    comparison_rows: list[
        dict[str, Any]
    ] = []

    for method_name, input_path in INPUT_FILES.items():
        row = read_first_csv_row(
            input_path
        )

        comparison_rows.append(
            {
                "retrieval_method": method_name,
                "number_of_questions": int(
                    row["number_of_questions"]
                ),
                "number_of_indexed_chunks": int(
                    row["number_of_indexed_chunks"]
                ),
                "evidence_recall_at_1": to_float(
                    row["evidence_recall_at_1"]
                ),
                "evidence_recall_at_3": to_float(
                    row["evidence_recall_at_3"]
                ),
                "evidence_recall_at_5": to_float(
                    row["evidence_recall_at_5"]
                ),
                "evidence_recall_at_10": to_float(
                    row["evidence_recall_at_10"]
                ),
                "evidence_mrr": to_float(
                    row["evidence_mrr"]
                ),
                "source_recall_at_5": to_float(
                    row["source_recall_at_5"]
                ),
                "source_mrr": to_float(
                    row["source_mrr"]
                ),
                "mean_retrieval_time_ms": to_float(
                    row["mean_retrieval_time_ms"]
                ),
                "p95_retrieval_time_ms": to_float(
                    row["p95_retrieval_time_ms"]
                ),
            }
        )

    comparison_rows.sort(
        key=lambda record: (
            record["evidence_mrr"],
            record["evidence_recall_at_1"],
            record["evidence_recall_at_5"],
        ),
        reverse=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                comparison_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    print("=" * 90)
    print("COMPARAISON DES RETRIEVERS")
    print("=" * 90)

    for rank, row in enumerate(
        comparison_rows,
        start=1,
    ):
        print(
            f"{rank}. {row['retrieval_method']} | "
            f"Recall@1={row['evidence_recall_at_1']:.4f} | "
            f"Recall@5={row['evidence_recall_at_5']:.4f} | "
            f"MRR={row['evidence_mrr']:.4f} | "
            f"Temps={row['mean_retrieval_time_ms']:.4f} ms"
        )

    print()
    print(
        "Tableau enregistré :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()