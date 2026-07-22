import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.config import ConfigLoader
from src.ingestion.discovery import discover_connectors
from src.ingestion.registry import ConnectorRegistry
from src.models.source import SourceConfig


SOURCE_ID = "azure_well_architected_github"


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("octets", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} octets"


def main() -> None:
    loader = ConfigLoader(PROJECT_ROOT)

    sources_config = loader.load_sources()
    ingestion_config = loader.load_ingestion()

    imported_modules = discover_connectors()

    configured_sources = [
        SourceConfig.from_dict(source_data)
        for source_data in sources_config["sources"]
    ]

    github_source = next(
        (
            source
            for source in configured_sources
            if source.id == SOURCE_ID
        ),
        None,
    )

    if github_source is None:
        raise RuntimeError(
            f"Source introuvable dans sources.yaml : "
            f"{SOURCE_ID}"
        )

    print("=" * 70)
    print("TEST DU GITHUB CONNECTOR")
    print("=" * 70)

    print(
        f"Modules découverts : "
        f"{len(imported_modules)}"
    )

    print(
        "Connecteurs disponibles : "
        + ", ".join(
            ConnectorRegistry.available_connectors()
        )
    )

    connector = ConnectorRegistry.create(
        source=github_source,
        project_root=PROJECT_ROOT,
        ingestion_config=ingestion_config,
    )

    result = connector.ingest()

    print(f"\nSource  : {result.source_id}")
    print(f"Statut  : {result.status}")
    print(f"Fichiers: {result.file_count}")

    if result.errors:
        print("\nErreurs :")

        for error in result.errors:
            print(f"  - {error}")

        raise SystemExit(1)

    print(
        "Action  : "
        f"{result.metadata.get('action')}"
    )

    print(
        "Branche : "
        f"{result.metadata.get('branch')}"
    )

    print(
        "Commit  : "
        f"{result.metadata.get('commit_sha')}"
    )

    print(
        "Date    : "
        f"{result.metadata.get('commit_date')}"
    )

    print(
        "Taille  : "
        f"{format_size(result.metadata.get('total_size_bytes', 0))}"
    )

    print("\nExtensions :")

    for extension, count in result.metadata.get(
        "extension_counts",
        {},
    ).items():
        print(f"  - {extension}: {count}")

    print("\nPremiers fichiers :")

    for file_path in result.files[:5]:
        print(f"  - {file_path}")

    assert result.status in {
        "success",
        "empty",
    }

    assert result.metadata.get(
        "commit_sha"
    )

    print("\n GitHubConnector fonctionnel.")


if __name__ == "__main__":
    main()