import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.common.config import (
    ConfigLoader,
    ConfigurationError,
)
from src.ingestion.discovery import (
    ConnectorDiscoveryError,
    discover_connectors,
)
from src.ingestion.registry import (
    ConnectorRegistry,
    ConnectorRegistryError,
)
from src.models.source import (
    SourceConfig,
    SourceConfigurationError,
)


def format_size(size_bytes: int) -> str:
    """
    Affiche une taille de fichier de manière lisible.
    """

    size = float(size_bytes)

    for unit in ("octets", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} octets"


def main() -> None:
    try:
        loader = ConfigLoader(PROJECT_ROOT)

        sources_config = loader.load_sources()
        ingestion_config = loader.load_ingestion()

        imported_modules = discover_connectors()

        print("=" * 70)
        print("AZURE RAG — TEST DU FRAMEWORK DE CONNECTEURS")
        print("=" * 70)

        print(
            f"Modules de connecteurs découverts : "
            f"{len(imported_modules)}"
        )

        print(
            "Connecteurs enregistrés : "
            + ", ".join(
                ConnectorRegistry.available_connectors()
            )
        )

        configured_sources = [
            SourceConfig.from_dict(source_data)
            for source_data in sources_config["sources"]
        ]

        enabled_sources = [
            source
            for source in configured_sources
            if source.enabled
        ]

        print(
            f"Sources activées : {len(enabled_sources)}\n"
        )

        success_count = 0
        failure_count = 0

        for source in enabled_sources:
            print("-" * 70)
            print(f"Source : {source.name}")
            print(f"ID     : {source.id}")
            print(f"Type   : {source.type}")

            try:
                connector = ConnectorRegistry.create(
                    source=source,
                    project_root=PROJECT_ROOT,
                    ingestion_config=ingestion_config,
                )

                result = connector.ingest()

                print(f"Statut : {result.status}")
                print(
                    f"Fichiers trouvés : "
                    f"{result.file_count}"
                )

                total_size = result.metadata.get(
                    "total_size_bytes",
                    0,
                )

                print(
                    f"Taille totale : "
                    f"{format_size(total_size)}"
                )

                extension_counts = result.metadata.get(
                    "extension_counts",
                    {},
                )

                if extension_counts:
                    print("Extensions :")

                    for extension, count in extension_counts.items():
                        print(
                            f"  - {extension}: {count}"
                        )

                if result.errors:
                    print("Erreurs :")

                    for error in result.errors:
                        print(f"  - {error}")

                if result.success:
                    success_count += 1
                else:
                    failure_count += 1

            except ConnectorRegistryError as error:
                failure_count += 1
                print(f"Statut : connecteur indisponible")
                print(f"Erreur : {error}")

        print("\n" + "=" * 70)
        print("RÉSUMÉ")
        print("=" * 70)
        print(f"Sources réussies : {success_count}")
        print(f"Sources non traitées ou en erreur : {failure_count}")

    except (
        ConfigurationError,
        ConnectorDiscoveryError,
        ConnectorRegistryError,
        SourceConfigurationError,
    ) as error:
        print(f"Erreur : {error}")
        raise SystemExit(1)

    except Exception as error:
        print(f"Erreur inattendue : {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()