from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "fixed_1024_chunks.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "08_confidence_gate_technical_ooc"
)

ALL_RESULTS_PATH = (
    OUTPUT_DIR
    / "technology_presence_results.csv"
)

ABSENT_RESULTS_PATH = (
    OUTPUT_DIR
    / "absent_technologies.csv"
)

PRESENT_RESULTS_PATH = (
    OUTPUT_DIR
    / "present_technologies.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "technology_scan_manifest.json"
)


@dataclass(frozen=True)
class TechnologyCandidate:
    """
    Technologie candidate pour le Benchmark 2.

    aliases contient les variantes textuelles recherchées
    dans le corpus.
    """

    key: str
    display_name: str
    category: str
    aliases: tuple[str, ...]


TECHNOLOGIES: tuple[TechnologyCandidate, ...] = (
    # ========================================================
    # Réseaux et sécurité
    # ========================================================
    TechnologyCandidate(
        key="cisco_ios",
        display_name="Cisco IOS",
        category="networking",
        aliases=(
            "Cisco IOS",
            "IOS XE",
            "Cisco Internetwork Operating System",
        ),
    ),
    TechnologyCandidate(
        key="juniper_junos",
        display_name="Juniper JunOS",
        category="networking",
        aliases=(
            "Juniper JunOS",
            "JunOS",
            "Junos OS",
        ),
    ),
    TechnologyCandidate(
        key="mikrotik_routeros",
        display_name="MikroTik RouterOS",
        category="networking",
        aliases=(
            "MikroTik",
            "RouterOS",
        ),
    ),
    TechnologyCandidate(
        key="fortigate",
        display_name="FortiGate",
        category="network_security",
        aliases=(
            "FortiGate",
            "Fortinet FortiGate",
            "FortiOS",
        ),
    ),
    TechnologyCandidate(
        key="pfsense",
        display_name="pfSense",
        category="network_security",
        aliases=(
            "pfSense",
        ),
    ),

    # ========================================================
    # Virtualisation
    # ========================================================
    TechnologyCandidate(
        key="vmware_esxi",
        display_name="VMware ESXi",
        category="virtualization",
        aliases=(
            "VMware ESXi",
            "ESXi",
        ),
    ),
    TechnologyCandidate(
        key="vmware_vcenter",
        display_name="VMware vCenter",
        category="virtualization",
        aliases=(
            "VMware vCenter",
            "vCenter Server",
            "vCenter",
        ),
    ),
    TechnologyCandidate(
        key="proxmox_ve",
        display_name="Proxmox VE",
        category="virtualization",
        aliases=(
            "Proxmox VE",
            "Proxmox Virtual Environment",
            "Proxmox",
        ),
    ),
    TechnologyCandidate(
        key="xenserver",
        display_name="XenServer",
        category="virtualization",
        aliases=(
            "XenServer",
            "Citrix Hypervisor",
        ),
    ),
    TechnologyCandidate(
        key="hyper_v_replica",
        display_name="Hyper-V Replica",
        category="virtualization",
        aliases=(
            "Hyper-V Replica",
            "Hyper V Replica",
        ),
    ),

    # ========================================================
    # DevOps et automatisation
    # ========================================================
    TechnologyCandidate(
        key="jenkins",
        display_name="Jenkins",
        category="devops",
        aliases=(
            "Jenkins",
            "Jenkins Pipeline",
        ),
    ),
    TechnologyCandidate(
        key="gitlab_cicd",
        display_name="GitLab CI/CD",
        category="devops",
        aliases=(
            "GitLab CI/CD",
            "GitLab CI",
            "GitLab Pipeline",
        ),
    ),
    TechnologyCandidate(
        key="teamcity",
        display_name="TeamCity",
        category="devops",
        aliases=(
            "TeamCity",
        ),
    ),
    TechnologyCandidate(
        key="bamboo",
        display_name="Atlassian Bamboo",
        category="devops",
        aliases=(
            "Atlassian Bamboo",
            "Bamboo CI",
        ),
    ),
    TechnologyCandidate(
        key="argocd",
        display_name="Argo CD",
        category="devops",
        aliases=(
            "Argo CD",
            "ArgoCD",
        ),
    ),

    # ========================================================
    # Bases de données
    # ========================================================
    TechnologyCandidate(
        key="oracle_database",
        display_name="Oracle Database",
        category="database",
        aliases=(
            "Oracle Database",
            "Oracle DB",
            "Oracle RAC",
            "Oracle Data Guard",
        ),
    ),
    TechnologyCandidate(
        key="mariadb_galera",
        display_name="MariaDB Galera Cluster",
        category="database",
        aliases=(
            "MariaDB Galera",
            "Galera Cluster",
        ),
    ),
    TechnologyCandidate(
        key="ibm_db2",
        display_name="IBM Db2",
        category="database",
        aliases=(
            "IBM Db2",
            "IBM DB2",
            "DB2 Database",
        ),
    ),
    TechnologyCandidate(
        key="sap_hana",
        display_name="SAP HANA",
        category="database",
        aliases=(
            "SAP HANA",
            "HANA Database",
        ),
    ),
    TechnologyCandidate(
        key="postgresql_streaming_replication",
        display_name="PostgreSQL Streaming Replication",
        category="database",
        aliases=(
            "PostgreSQL Streaming Replication",
            "Postgres Streaming Replication",
        ),
    ),

    # ========================================================
    # Administration système
    # ========================================================
    TechnologyCandidate(
        key="red_hat_satellite",
        display_name="Red Hat Satellite",
        category="system_administration",
        aliases=(
            "Red Hat Satellite",
            "Satellite Server",
        ),
    ),
    TechnologyCandidate(
        key="ansible_tower",
        display_name="Ansible Tower",
        category="automation",
        aliases=(
            "Ansible Tower",
            "Automation Controller",
        ),
    ),
    TechnologyCandidate(
        key="puppet_enterprise",
        display_name="Puppet Enterprise",
        category="automation",
        aliases=(
            "Puppet Enterprise",
        ),
    ),
    TechnologyCandidate(
        key="chef_infra",
        display_name="Chef Infra",
        category="automation",
        aliases=(
            "Chef Infra",
            "Chef Server",
        ),
    ),

    # ========================================================
    # Conteneurs et orchestration
    # ========================================================
    TechnologyCandidate(
        key="docker_swarm",
        display_name="Docker Swarm",
        category="containers",
        aliases=(
            "Docker Swarm",
            "Swarm Mode",
        ),
    ),
    TechnologyCandidate(
        key="openshift",
        display_name="Red Hat OpenShift",
        category="containers",
        aliases=(
            "OpenShift",
            "Red Hat OpenShift",
        ),
    ),
    TechnologyCandidate(
        key="rancher",
        display_name="Rancher",
        category="containers",
        aliases=(
            "Rancher",
            "Rancher Kubernetes Engine",
            "RKE2",
        ),
    ),
    TechnologyCandidate(
        key="hashicorp_nomad",
        display_name="HashiCorp Nomad",
        category="containers",
        aliases=(
            "HashiCorp Nomad",
            "Nomad Cluster",
        ),
    ),
    TechnologyCandidate(
        key="podman",
        display_name="Podman",
        category="containers",
        aliases=(
            "Podman",
        ),
    ),

    # ========================================================
    # Supervision et observabilité
    # ========================================================
    TechnologyCandidate(
        key="zabbix",
        display_name="Zabbix",
        category="monitoring",
        aliases=(
            "Zabbix",
        ),
    ),
    TechnologyCandidate(
        key="nagios",
        display_name="Nagios",
        category="monitoring",
        aliases=(
            "Nagios",
            "Nagios Core",
        ),
    ),
    TechnologyCandidate(
        key="solarwinds",
        display_name="SolarWinds",
        category="monitoring",
        aliases=(
            "SolarWinds",
            "SolarWinds Orion",
        ),
    ),
    TechnologyCandidate(
        key="graylog",
        display_name="Graylog",
        category="logging",
        aliases=(
            "Graylog",
        ),
    ),
    TechnologyCandidate(
        key="splunk_enterprise",
        display_name="Splunk Enterprise",
        category="logging",
        aliases=(
            "Splunk Enterprise",
            "Splunk Indexer",
        ),
    ),
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def normalize_text(text: Any) -> str:
    """
    Normalisation légère destinée à la recherche
    de mots-clés dans le corpus.
    """

    value = str(
        text if text is not None else ""
    )

    value = value.casefold()

    value = re.sub(
        r"[\s_\-/\\]+",
        " ",
        value,
    )

    value = re.sub(
        r"[^\w\s.+#]",
        " ",
        value,
    )

    value = " ".join(
        value.split()
    )

    return value


def extract_searchable_text(
    record: dict[str, Any],
) -> str:
    """
    Construit un texte de recherche à partir des champs
    possibles d'un chunk.

    Le script supporte plusieurs schémas JSONL :
    - content
    - text
    - page_content
    - chunk_text
    - source
    - title
    - metadata
    """

    values: list[str] = []

    direct_fields = (
        "content",
        "text",
        "page_content",
        "chunk_text",
        "source",
        "title",
        "document_format",
        "chunk_id",
    )

    for field_name in direct_fields:
        value = record.get(
            field_name
        )

        if value is not None:
            values.append(
                str(value)
            )

    metadata = record.get(
        "metadata"
    )

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            values.append(
                str(key)
            )

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                values.append(
                    str(value)
                )

            elif isinstance(
                value,
                list,
            ):
                values.extend(
                    str(item)
                    for item in value
                )

    return normalize_text(
        "\n".join(values)
    )


def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "JSON invalide à la ligne "
                    f"{line_number}."
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "Chaque ligne doit contenir "
                    f"un objet JSON. Ligne {line_number}."
                )

            chunks.append(record)

    if not chunks:
        raise ValueError(
            "Aucun chunk trouvé dans le fichier."
        )

    return chunks


def alias_is_present(
    normalized_corpus_text: str,
    alias: str,
) -> bool:
    normalized_alias = normalize_text(
        alias
    )

    if not normalized_alias:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(
            normalized_alias
        )
        + r"(?!\w)"
    )

    return (
        re.search(
            pattern,
            normalized_corpus_text,
        )
        is not None
    )


def scan_technology(
    technology: TechnologyCandidate,
    searchable_chunks: list[
        tuple[dict[str, Any], str]
    ],
) -> dict[str, Any]:
    matching_chunks: list[
        dict[str, Any]
    ] = []

    matched_aliases: set[str] = set()

    for record, searchable_text in (
        searchable_chunks
    ):
        aliases_found = [
            alias
            for alias in technology.aliases
            if alias_is_present(
                searchable_text,
                alias,
            )
        ]

        if not aliases_found:
            continue

        matched_aliases.update(
            aliases_found
        )

        matching_chunks.append(
            record
        )

    example_sources: list[str] = []

    for record in matching_chunks:
        source = str(
            record.get(
                "source",
                "",
            )
            or (
                record.get(
                    "metadata",
                    {},
                ).get(
                    "source",
                    ""
                )
                if isinstance(
                    record.get(
                        "metadata"
                    ),
                    dict,
                )
                else ""
            )
        ).strip()

        if (
            source
            and source
            not in example_sources
        ):
            example_sources.append(
                source
            )

        if len(example_sources) >= 5:
            break

    return {
        "technology_key": technology.key,
        "technology": (
            technology.display_name
        ),
        "category": technology.category,
        "aliases_checked": " | ".join(
            technology.aliases
        ),
        "present_in_corpus": (
            len(matching_chunks) > 0
        ),
        "matching_chunks": len(
            matching_chunks
        ),
        "matched_aliases": " | ".join(
            sorted(matched_aliases)
        ),
        "example_sources": " | ".join(
            example_sources
        ),
    }


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
    print(
        "VÉRIFICATION DES TECHNOLOGIES "
        "INFORMATIQUES HORS CORPUS"
    )
    print("=" * 100)

    require_file(
        CHUNKS_PATH
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    print(
        "Chunks chargés :",
        len(chunks),
    )

    searchable_chunks = [
        (
            record,
            extract_searchable_text(
                record
            ),
        )
        for record in chunks
    ]

    results: list[
        dict[str, Any]
    ] = []

    for index, technology in enumerate(
        TECHNOLOGIES,
        start=1,
    ):
        result = scan_technology(
            technology=technology,
            searchable_chunks=(
                searchable_chunks
            ),
        )

        results.append(
            result
        )

        status = (
            "PRÉSENTE"
            if result[
                "present_in_corpus"
            ]
            else "ABSENTE"
        )

        print(
            f"[{index:02d}/"
            f"{len(TECHNOLOGIES):02d}] "
            f"{technology.display_name:<35} "
            f"| {status:<8} "
            f"| chunks : "
            f"{result['matching_chunks']}"
        )

    absent_results = [
        row
        for row in results
        if not row[
            "present_in_corpus"
        ]
    ]

    present_results = [
        row
        for row in results
        if row[
            "present_in_corpus"
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        ALL_RESULTS_PATH,
        results,
    )

    write_csv(
        ABSENT_RESULTS_PATH,
        absent_results,
    )

    write_csv(
        PRESENT_RESULTS_PATH,
        present_results,
    )

    manifest = {
        "benchmark_name": (
            "technical_out_of_corpus_scan"
        ),
        "chunks_path": str(
            CHUNKS_PATH
        ),
        "number_of_chunks": len(
            chunks
        ),
        "number_of_candidates": len(
            TECHNOLOGIES
        ),
        "number_of_absent_technologies": (
            len(absent_results)
        ),
        "number_of_present_technologies": (
            len(present_results)
        ),
        "search_method": (
            "case-insensitive exact alias matching "
            "over content, source, title and metadata"
        ),
        "files": {
            "all_results": str(
                ALL_RESULTS_PATH
            ),
            "absent_results": str(
                ABSENT_RESULTS_PATH
            ),
            "present_results": str(
                PRESENT_RESULTS_PATH
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
    print("RÉSUMÉ")
    print("=" * 100)

    print(
        "Technologies candidates :",
        len(TECHNOLOGIES),
    )

    print(
        "Technologies absentes :",
        len(absent_results),
    )

    print(
        "Technologies présentes :",
        len(present_results),
    )

    print()
    print(
        "Résultats complets :",
        ALL_RESULTS_PATH,
    )
    print(
        "Technologies absentes :",
        ABSENT_RESULTS_PATH,
    )
    print(
        "Technologies présentes :",
        PRESENT_RESULTS_PATH,
    )
    print(
        "Manifeste :",
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()