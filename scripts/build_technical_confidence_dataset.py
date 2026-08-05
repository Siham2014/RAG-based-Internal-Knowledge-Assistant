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

ABSENT_TECHNOLOGIES_PATH = (
    PROJECT_ROOT
    / "benchmarking"
    / "08_confidence_gate_technical_ooc"
    / "absent_technologies.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "08_confidence_gate_technical_ooc"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "confidence_gate_technical_questions.csv"
)


OUTPUT_COLUMNS = [
    "id",
    "question",
    "expected_answerable",
    "category",
    "technology",
    "expected_source",
    "gold_evidence",
    "label_reason",
]


TECHNICAL_UNANSWERABLE_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "T001",
        "question": (
            "How do I configure a VLAN on a Cisco IOS switch?"
        ),
        "technology": "Cisco IOS",
        "category": "technical_networking",
    },
    {
        "id": "T002",
        "question": (
            "How do I configure OSPF routing on Juniper JunOS?"
        ),
        "technology": "Juniper JunOS",
        "category": "technical_networking",
    },
    {
        "id": "T003",
        "question": (
            "How do I create a firewall rule in MikroTik RouterOS?"
        ),
        "technology": "MikroTik RouterOS",
        "category": "technical_networking",
    },
    {
        "id": "T004",
        "question": (
            "How do I configure an IPsec VPN on a FortiGate firewall?"
        ),
        "technology": "FortiGate",
        "category": "technical_network_security",
    },
    {
        "id": "T005",
        "question": (
            "How do I configure a site-to-site VPN with pfSense?"
        ),
        "technology": "pfSense",
        "category": "technical_network_security",
    },
    {
        "id": "T006",
        "question": (
            "How do I install VMware ESXi on a physical server?"
        ),
        "technology": "VMware ESXi",
        "category": "technical_virtualization",
    },
    {
        "id": "T007",
        "question": (
            "How do I add an ESXi host to VMware vCenter?"
        ),
        "technology": "VMware vCenter",
        "category": "technical_virtualization",
    },
    {
        "id": "T008",
        "question": (
            "How do I create a virtual machine in Proxmox VE?"
        ),
        "technology": "Proxmox VE",
        "category": "technical_virtualization",
    },
    {
        "id": "T009",
        "question": (
            "How do I create a virtual machine in XenServer?"
        ),
        "technology": "XenServer",
        "category": "technical_virtualization",
    },
    {
        "id": "T010",
        "question": (
            "How do I configure Hyper-V Replica between two hosts?"
        ),
        "technology": "Hyper-V Replica",
        "category": "technical_virtualization",
    },
    {
        "id": "T011",
        "question": (
            "How do I create a declarative pipeline in Jenkins?"
        ),
        "technology": "Jenkins",
        "category": "technical_devops",
    },
    {
        "id": "T012",
        "question": (
            "How do I register a runner for GitLab CI/CD?"
        ),
        "technology": "GitLab CI/CD",
        "category": "technical_devops",
    },
    {
        "id": "T013",
        "question": (
            "How do I configure a build agent in TeamCity?"
        ),
        "technology": "TeamCity",
        "category": "technical_devops",
    },
    {
        "id": "T014",
        "question": (
            "How do I create a deployment plan in Atlassian Bamboo?"
        ),
        "technology": "Atlassian Bamboo",
        "category": "technical_devops",
    },
    {
        "id": "T015",
        "question": (
            "How do I synchronize an application with Argo CD?"
        ),
        "technology": "Argo CD",
        "category": "technical_devops",
    },
    {
        "id": "T016",
        "question": (
            "How do I configure a MariaDB Galera Cluster?"
        ),
        "technology": "MariaDB Galera Cluster",
        "category": "technical_database",
    },
    {
        "id": "T017",
        "question": (
            "How do I create a database and table space in IBM Db2?"
        ),
        "technology": "IBM Db2",
        "category": "technical_database",
    },
    {
        "id": "T018",
        "question": (
            "How do I configure PostgreSQL streaming replication?"
        ),
        "technology": "PostgreSQL Streaming Replication",
        "category": "technical_database",
    },
    {
        "id": "T019",
        "question": (
            "How do I register a Linux server with Red Hat Satellite?"
        ),
        "technology": "Red Hat Satellite",
        "category": "technical_system_administration",
    },
    {
        "id": "T020",
        "question": (
            "How do I create and launch a job template in Ansible Tower?"
        ),
        "technology": "Ansible Tower",
        "category": "technical_automation",
    },
    {
        "id": "T021",
        "question": (
            "How do I deploy an agent with Puppet Enterprise?"
        ),
        "technology": "Puppet Enterprise",
        "category": "technical_automation",
    },
    {
        "id": "T022",
        "question": (
            "How do I bootstrap a node with Chef Infra?"
        ),
        "technology": "Chef Infra",
        "category": "technical_automation",
    },
    {
        "id": "T023",
        "question": (
            "How do I create a project in Red Hat OpenShift?"
        ),
        "technology": "Red Hat OpenShift",
        "category": "technical_containers",
    },
    {
        "id": "T024",
        "question": (
            "How do I import a Kubernetes cluster into Rancher?"
        ),
        "technology": "Rancher",
        "category": "technical_containers",
    },
    {
        "id": "T025",
        "question": (
            "How do I submit a job to a HashiCorp Nomad cluster?"
        ),
        "technology": "HashiCorp Nomad",
        "category": "technical_containers",
    },
    {
        "id": "T026",
        "question": (
            "How do I build and run a container using Podman?"
        ),
        "technology": "Podman",
        "category": "technical_containers",
    },
    {
        "id": "T027",
        "question": (
            "How do I configure a host for monitoring in Zabbix?"
        ),
        "technology": "Zabbix",
        "category": "technical_monitoring",
    },
    {
        "id": "T028",
        "question": (
            "How do I define a service check in Nagios Core?"
        ),
        "technology": "Nagios",
        "category": "technical_monitoring",
    },
    {
        "id": "T029",
        "question": (
            "How do I create a log input in Graylog?"
        ),
        "technology": "Graylog",
        "category": "technical_logging",
    },
    {
        "id": "T030",
        "question": (
            "How do I create an index and ingest logs "
            "into Splunk Enterprise?"
        ),
        "technology": "Splunk Enterprise",
        "category": "technical_logging",
    },
]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_absent_technologies(
    path: Path,
) -> set[str]:
    technologies: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le fichier des technologies absentes "
                "ne contient pas d'en-tête."
            )

        if "technology" not in reader.fieldnames:
            raise ValueError(
                "La colonne 'technology' est absente."
            )

        for row in reader:
            technology = str(
                row.get("technology", "")
            ).strip()

            if technology:
                technologies.add(technology)

    if not technologies:
        raise ValueError(
            "Aucune technologie absente n'a été trouvée."
        )

    return technologies


def load_answerable_questions(
    path: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Le CSV des questions répondables "
                "ne contient pas d'en-tête."
            )

        required_columns = {
            "id",
            "question",
            "expected_source",
            "gold_evidence",
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Colonnes manquantes : "
                + ", ".join(sorted(missing))
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
                row.get("expected_source", "")
            ).strip()

            gold_evidence = str(
                row.get("gold_evidence", "")
            ).strip()

            if not question_id or not question:
                raise ValueError(
                    f"Donnée invalide à la ligne {row_number}."
                )

            records.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_answerable": "true",
                    "category": "azure_answerable",
                    "technology": "Microsoft Azure",
                    "expected_source": expected_source,
                    "gold_evidence": gold_evidence,
                    "label_reason": (
                        "Une source et une preuve de référence "
                        "existent dans le corpus."
                    ),
                }
            )

    if len(records) != 30:
        raise ValueError(
            "30 questions répondables sont attendues, "
            f"{len(records)} trouvées."
        )

    return records


def build_unanswerable_questions(
    absent_technologies: set[str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for item in TECHNICAL_UNANSWERABLE_QUESTIONS:
        technology = item["technology"]

        if technology not in absent_technologies:
            raise ValueError(
                "La technologie n'est pas confirmée comme "
                f"absente du corpus : {technology}"
            )

        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_answerable": "false",
                "category": item["category"],
                "technology": technology,
                "expected_source": "",
                "gold_evidence": "",
                "label_reason": (
                    "La technologie a été recherchée dans les "
                    "1 478 chunks et aucune occurrence n'a été trouvée."
                ),
            }
        )

    if len(records) != 30:
        raise ValueError(
            "30 questions techniques hors corpus sont "
            f"attendues, {len(records)} trouvées."
        )

    return records


def validate_dataset(
    records: list[dict[str, str]],
) -> None:
    if len(records) != 60:
        raise ValueError(
            f"60 questions attendues, {len(records)} trouvées."
        )

    identifiers: set[str] = set()
    questions: set[str] = set()

    answerable_count = 0
    unanswerable_count = 0

    for record in records:
        question_id = record["id"].strip()
        question = record["question"].strip()

        normalized_question = " ".join(
            question.casefold().split()
        )

        if question_id in identifiers:
            raise ValueError(
                f"ID dupliqué : {question_id}"
            )

        if normalized_question in questions:
            raise ValueError(
                f"Question dupliquée : {question}"
            )

        identifiers.add(question_id)
        questions.add(normalized_question)

        label = record["expected_answerable"]

        if label == "true":
            answerable_count += 1

            if not record["expected_source"].strip():
                raise ValueError(
                    f"Source absente pour {question_id}."
                )

            if not record["gold_evidence"].strip():
                raise ValueError(
                    f"Preuve absente pour {question_id}."
                )

        elif label == "false":
            unanswerable_count += 1

        else:
            raise ValueError(
                f"Label invalide pour {question_id}: {label}"
            )

    if answerable_count != 30:
        raise ValueError(
            f"30 répondables attendues, {answerable_count} trouvées."
        )

    if unanswerable_count != 30:
        raise ValueError(
            "30 non répondables attendues, "
            f"{unanswerable_count} trouvées."
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


def count_categories(
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
    print("=" * 100)
    print(
        "CONSTRUCTION DU BENCHMARK 2 — "
        "INFORMATIQUE HORS CORPUS"
    )
    print("=" * 100)

    require_file(
        ANSWERABLE_DATASET_PATH
    )

    require_file(
        ABSENT_TECHNOLOGIES_PATH
    )

    absent_technologies = (
        load_absent_technologies(
            ABSENT_TECHNOLOGIES_PATH
        )
    )

    answerable_records = (
        load_answerable_questions(
            ANSWERABLE_DATASET_PATH
        )
    )

    unanswerable_records = (
        build_unanswerable_questions(
            absent_technologies
        )
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

    category_counts = count_categories(
        final_records
    )

    print(
        "Technologies confirmées absentes :",
        len(absent_technologies),
    )

    print(
        "Questions répondables :",
        len(answerable_records),
    )

    print(
        "Questions techniques hors corpus :",
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
        print(f"- {category}: {count}")

    print()
    print(
        "Dataset sauvegardé :",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()