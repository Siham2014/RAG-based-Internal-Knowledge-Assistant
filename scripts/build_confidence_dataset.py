from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANSWERABLE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_30_questions.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "benchmarking"
    / "07_confidence_gate_benchmark"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "confidence_gate_questions.csv"
)


UNANSWERABLE_QUESTIONS: list[dict[str, str]] = [
    # --------------------------------------------------------
    # Culture générale — clairement hors du corpus Azure
    # --------------------------------------------------------
    {
        "id": "U001",
        "question": (
            "Who painted the Mona Lisa?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question artistique sans relation avec "
            "la documentation Azure."
        ),
    },
    {
        "id": "U002",
        "question": (
            "What is the capital city of Brazil?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question géographique hors domaine."
        ),
    },
    {
        "id": "U003",
        "question": (
            "Who wrote the novel Pride and Prejudice?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question littéraire hors domaine."
        ),
    },
    {
        "id": "U004",
        "question": (
            "Which country won the 2022 FIFA World Cup?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question sportive hors domaine."
        ),
    },
    {
        "id": "U005",
        "question": (
            "What is the tallest mountain in Africa?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question géographique hors domaine."
        ),
    },
    {
        "id": "U006",
        "question": (
            "Who composed The Four Seasons?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question musicale hors domaine."
        ),
    },
    {
        "id": "U007",
        "question": (
            "What language is primarily spoken in Argentina?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question linguistique hors domaine."
        ),
    },
    {
        "id": "U008",
        "question": (
            "In which century was the Eiffel Tower built?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question historique hors domaine."
        ),
    },
    {
        "id": "U009",
        "question": (
            "What is the largest ocean on Earth?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question géographique hors domaine."
        ),
    },
    {
        "id": "U010",
        "question": (
            "Who was the first person to walk on the Moon?"
        ),
        "category": "general_knowledge",
        "label_reason": (
            "Question historique hors domaine."
        ),
    },

    # --------------------------------------------------------
    # Sciences naturelles et vie quotidienne
    # --------------------------------------------------------
    {
        "id": "U011",
        "question": (
            "How does photosynthesis work in green plants?"
        ),
        "category": "natural_science",
        "label_reason": (
            "Sujet de biologie absent du corpus technique."
        ),
    },
    {
        "id": "U012",
        "question": (
            "What is the function of red blood cells?"
        ),
        "category": "natural_science",
        "label_reason": (
            "Sujet médical et biologique hors domaine."
        ),
    },
    {
        "id": "U013",
        "question": (
            "Why do earthquakes occur?"
        ),
        "category": "natural_science",
        "label_reason": (
            "Sujet géologique hors domaine."
        ),
    },
    {
        "id": "U014",
        "question": (
            "What causes ocean tides?"
        ),
        "category": "natural_science",
        "label_reason": (
            "Sujet de physique naturelle hors domaine."
        ),
    },
    {
        "id": "U015",
        "question": (
            "How do bees produce honey?"
        ),
        "category": "natural_science",
        "label_reason": (
            "Sujet biologique hors domaine."
        ),
    },
    {
        "id": "U016",
        "question": (
            "What ingredients are needed to make couscous?"
        ),
        "category": "daily_life",
        "label_reason": (
            "Question culinaire hors domaine."
        ),
    },
    {
        "id": "U017",
        "question": (
            "How should a tomato plant be watered?"
        ),
        "category": "daily_life",
        "label_reason": (
            "Question de jardinage hors domaine."
        ),
    },
    {
        "id": "U018",
        "question": (
            "What is the traditional recipe for Moroccan mint tea?"
        ),
        "category": "daily_life",
        "label_reason": (
            "Question culinaire hors domaine."
        ),
    },
    {
        "id": "U019",
        "question": (
            "How long should pasta be cooked?"
        ),
        "category": "daily_life",
        "label_reason": (
            "Question culinaire hors domaine."
        ),
    },
    {
        "id": "U020",
        "question": (
            "What is the best season for growing strawberries?"
        ),
        "category": "daily_life",
        "label_reason": (
            "Question agricole hors domaine."
        ),
    },

    # --------------------------------------------------------
    # Informations internes fictives d’entreprises
    # Ces questions sont proches d’un futur cas d’usage RAG,
    # mais ne doivent pas être répondues sans documents.
    # --------------------------------------------------------
    {
        "id": "U021",
        "question": (
            "What is Northwind Aerospace's annual "
            "travel reimbursement limit?"
        ),
        "category": "unsupported_company_policy",
        "label_reason": (
            "Politique interne fictive absente du corpus."
        ),
    },
    {
        "id": "U022",
        "question": (
            "How many paid leave days does Fabrikam Mining "
            "grant to new employees?"
        ),
        "category": "unsupported_company_policy",
        "label_reason": (
            "Règle RH interne fictive absente du corpus."
        ),
    },
    {
        "id": "U023",
        "question": (
            "Who is the current safety manager at "
            "Adventure Works Manufacturing?"
        ),
        "category": "unsupported_company_fact",
        "label_reason": (
            "Information organisationnelle fictive "
            "absente du corpus."
        ),
    },
    {
        "id": "U024",
        "question": (
            "What is the emergency phone number for "
            "Contoso Chemicals Plant 4?"
        ),
        "category": "unsupported_company_fact",
        "label_reason": (
            "Coordonnée interne non présente dans "
            "la documentation évaluée."
        ),
    },
    {
        "id": "U025",
        "question": (
            "What approval process does Alpine Logistics use "
            "for purchases above 10,000 euros?"
        ),
        "category": "unsupported_company_policy",
        "label_reason": (
            "Processus financier interne fictif."
        ),
    },
    {
        "id": "U026",
        "question": (
            "Which protective gloves are mandatory in "
            "Wingtip Energy's laboratory?"
        ),
        "category": "unsupported_company_policy",
        "label_reason": (
            "Consigne HSE propre à une entreprise fictive."
        ),
    },
    {
        "id": "U027",
        "question": (
            "What is the maintenance interval for conveyor "
            "CV-204 at Blue Yonder Mining?"
        ),
        "category": "unsupported_company_fact",
        "label_reason": (
            "Donnée de maintenance interne fictive."
        ),
    },
    {
        "id": "U028",
        "question": (
            "Where is the assembly point for employees at "
            "Litware Factory B?"
        ),
        "category": "unsupported_company_fact",
        "label_reason": (
            "Information d'urgence interne fictive."
        ),
    },
    {
        "id": "U029",
        "question": (
            "What password rotation period is required by "
            "Tailspin Toys' internal security policy?"
        ),
        "category": "unsupported_company_policy",
        "label_reason": (
            "Exigence de sécurité interne fictive."
        ),
    },
    {
        "id": "U030",
        "question": (
            "Which supplier provides hydraulic hoses for "
            "Proseware Industrial Services?"
        ),
        "category": "unsupported_company_fact",
        "label_reason": (
            "Information d'approvisionnement fictive "
            "absente du corpus."
        ),
    },
]


OUTPUT_COLUMNS = [
    "id",
    "question",
    "expected_answerable",
    "category",
    "expected_source",
    "gold_evidence",
    "label_reason",
]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def normalize_boolean_text(
    value: bool,
) -> str:
    return (
        "true"
        if value
        else "false"
    )


def load_answerable_questions(
    path: Path,
) -> list[dict[str, str]]:
    """
    Charge les 30 questions existantes et les marque
    comme répondables.
    """

    answerable_records: list[
        dict[str, str]
    ] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le CSV d'évaluation ne contient "
                "pas d'en-tête."
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
                "Colonnes manquantes dans le CSV : "
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

            expected_source = str(
                row.get(
                    "expected_source",
                    "",
                )
            ).strip()

            gold_evidence = str(
                row.get(
                    "gold_evidence",
                    "",
                )
            ).strip()

            if not question_id:
                raise ValueError(
                    f"ID vide à la ligne {row_number}."
                )

            if not question:
                raise ValueError(
                    "Question vide à la ligne "
                    f"{row_number}."
                )

            answerable_records.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_answerable": (
                        normalize_boolean_text(
                            True
                        )
                    ),
                    "category": (
                        "azure_answerable"
                    ),
                    "expected_source": (
                        expected_source
                    ),
                    "gold_evidence": (
                        gold_evidence
                    ),
                    "label_reason": (
                        "Une source et une preuve de "
                        "référence existent dans le corpus."
                    ),
                }
            )

    return answerable_records


def build_unanswerable_records(
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for item in UNANSWERABLE_QUESTIONS:
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_answerable": (
                    normalize_boolean_text(
                        False
                    )
                ),
                "category": item["category"],
                "expected_source": "",
                "gold_evidence": "",
                "label_reason": (
                    item["label_reason"]
                ),
            }
        )

    return records


def validate_dataset(
    records: list[dict[str, str]],
) -> None:
    if not records:
        raise ValueError(
            "Le dataset final est vide."
        )

    identifiers: set[str] = set()
    normalized_questions: set[str] = set()

    answerable_count = 0
    unanswerable_count = 0

    for index, record in enumerate(
        records,
        start=1,
    ):
        question_id = record["id"].strip()
        question = record["question"].strip()

        if question_id in identifiers:
            raise ValueError(
                f"ID dupliqué : {question_id}"
            )

        identifiers.add(question_id)

        normalized_question = (
            " ".join(
                question.lower().split()
            )
        )

        if (
            normalized_question
            in normalized_questions
        ):
            raise ValueError(
                "Question dupliquée : "
                f"{question}"
            )

        normalized_questions.add(
            normalized_question
        )

        label = record[
            "expected_answerable"
        ]

        if label == "true":
            answerable_count += 1
        elif label == "false":
            unanswerable_count += 1
        else:
            raise ValueError(
                "Label invalide pour "
                f"{question_id} : {label}"
            )

        if (
            label == "true"
            and not record[
                "expected_source"
            ].strip()
        ):
            raise ValueError(
                "Source attendue absente pour "
                f"{question_id}."
            )

        if (
            label == "true"
            and not record[
                "gold_evidence"
            ].strip()
        ):
            raise ValueError(
                "Preuve attendue absente pour "
                f"{question_id}."
            )

    if answerable_count != 30:
        raise ValueError(
            "30 questions répondables sont "
            f"attendues, {answerable_count} trouvées."
        )

    if unanswerable_count != 30:
        raise ValueError(
            "30 questions non répondables sont "
            f"attendues, {unanswerable_count} trouvées."
        )


def write_dataset(
    path: Path,
    records: list[dict[str, str]],
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
        writer.writerows(records)


def count_by_category(
    records: list[dict[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for record in records:
        category = record["category"]

        counts[category] = (
            counts.get(category, 0)
            + 1
        )

    return counts


def main() -> None:
    print("=" * 90)
    print(
        "CONSTRUCTION DU DATASET "
        "DU CONFIDENCE GATE"
    )
    print("=" * 90)

    require_file(
        ANSWERABLE_DATASET_PATH
    )

    answerable_records = (
        load_answerable_questions(
            ANSWERABLE_DATASET_PATH
        )
    )

    unanswerable_records = (
        build_unanswerable_records()
    )

    final_records = (
        answerable_records
        + unanswerable_records
    )

    validate_dataset(
        final_records
    )

    write_dataset(
        OUTPUT_PATH,
        final_records,
    )

    category_counts = (
        count_by_category(
            final_records
        )
    )

    print(
        "Questions répondables :",
        len(answerable_records),
    )

    print(
        "Questions non répondables :",
        len(unanswerable_records),
    )

    print(
        "Total :",
        len(final_records),
    )

    print()
    print("Répartition par catégorie :")

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"- {category}: {count}"
        )

    print()
    print(
        "Dataset sauvegardé :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()