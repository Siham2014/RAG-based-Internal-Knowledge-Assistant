from pathlib import Path
from typing import Any

from src.ingestion.connectors.base import (
    BaseConnector,
    ConnectorValidationError,
    IngestionResult,
)
from src.ingestion.registry import ConnectorRegistry


@ConnectorRegistry.register("local_folder")
class LocalFolderConnector(BaseConnector):
    """
    Connecteur chargé de découvrir les documents présents
    dans un dossier local.

    Il applique :
    - un filtre sur les extensions ;
    - une exclusion de certains répertoires ;
    - une exploration récursive ou non récursive.
    """

    def validate(self) -> None:
        input_config = self.source.input

        configured_path = input_config.get("path")

        if not configured_path:
            raise ConnectorValidationError(
                f"La source '{self.source.id}' doit définir input.path."
            )

        source_directory = self.resolve_path(configured_path)

        if not source_directory.exists():
            raise ConnectorValidationError(
                f"Le dossier source n'existe pas : {source_directory}"
            )

        if not source_directory.is_dir():
            raise ConnectorValidationError(
                f"Le chemin configuré n'est pas un dossier : "
                f"{source_directory}"
            )

        extensions = self.source.filters.get("extensions", [])

        if extensions and not isinstance(extensions, list):
            raise ConnectorValidationError(
                f"filters.extensions doit être une liste "
                f"pour la source '{self.source.id}'."
            )

        excluded_directories = self.source.filters.get(
            "exclude_directories",
            [],
        )

        if excluded_directories and not isinstance(
            excluded_directories,
            list,
        ):
            raise ConnectorValidationError(
                f"filters.exclude_directories doit être une liste "
                f"pour la source '{self.source.id}'."
            )

    def ingest(self) -> IngestionResult:
        try:
            self.validate()

            input_config = self.source.input
            filters = self.source.filters

            source_directory = self.resolve_path(
                input_config["path"]
            )

            recursive = bool(
                input_config.get("recursive", True)
            )

            allowed_extensions = self._normalize_extensions(
                filters.get("extensions", [])
            )

            excluded_directories = {
                str(directory_name).strip().lower()
                for directory_name in filters.get(
                    "exclude_directories",
                    [],
                )
                if str(directory_name).strip()
            }

            discovered_files = self._discover_files(
                source_directory=source_directory,
                recursive=recursive,
                allowed_extensions=allowed_extensions,
                excluded_directories=excluded_directories,
            )

            total_size_bytes = sum(
                file_path.stat().st_size
                for file_path in discovered_files
            )

            extension_counts = self._count_extensions(
                discovered_files
            )

            status = "success"

            if not discovered_files:
                status = "empty"

            return IngestionResult(
                source_id=self.source.id,
                status=status,
                files=discovered_files,
                metadata={
                    "connector": self.connector_type,
                    "source_directory": str(source_directory),
                    "recursive": recursive,
                    "file_count": len(discovered_files),
                    "total_size_bytes": total_size_bytes,
                    "extension_counts": extension_counts,
                    "source_metadata": self.source.metadata,
                },
            )

        except Exception as error:
            return IngestionResult(
                source_id=self.source.id,
                status="failed",
                errors=[str(error)],
                metadata={
                    "connector": self.connector_type,
                },
            )

    def _discover_files(
        self,
        source_directory: Path,
        recursive: bool,
        allowed_extensions: set[str],
        excluded_directories: set[str],
    ) -> list[Path]:
        """
        Découvre les fichiers en appliquant les règles de filtrage.
        """

        iterator = (
            source_directory.rglob("*")
            if recursive
            else source_directory.glob("*")
        )

        files: list[Path] = []

        for path in iterator:
            if not path.is_file():
                continue

            relative_parts = path.relative_to(
                source_directory
            ).parts[:-1]

            normalized_parent_directories = {
                part.lower()
                for part in relative_parts
            }

            if (
                normalized_parent_directories
                & excluded_directories
            ):
                continue

            if (
                allowed_extensions
                and path.suffix.lower() not in allowed_extensions
            ):
                continue

            files.append(path.resolve())

        return sorted(
            files,
            key=lambda file_path: str(file_path).lower(),
        )

    @staticmethod
    def _normalize_extensions(
        extensions: list[Any],
    ) -> set[str]:
        """
        Uniformise les extensions :
        md devient .md, PDF devient .pdf, etc.
        """

        normalized_extensions: set[str] = set()

        for extension in extensions:
            value = str(extension).strip().lower()

            if not value:
                continue

            if not value.startswith("."):
                value = f".{value}"

            normalized_extensions.add(value)

        return normalized_extensions

    @staticmethod
    def _count_extensions(
        files: list[Path],
    ) -> dict[str, int]:
        """
        Calcule le nombre de fichiers par extension.
        """

        counts: dict[str, int] = {}

        for file_path in files:
            extension = (
                file_path.suffix.lower()
                if file_path.suffix
                else "[sans extension]"
            )

            counts[extension] = counts.get(extension, 0) + 1

        return dict(sorted(counts.items()))