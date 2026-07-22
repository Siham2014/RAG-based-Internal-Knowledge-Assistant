from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.source import SourceConfig


class ConnectorError(Exception):
    """Erreur générale produite par un connecteur."""


class ConnectorValidationError(ConnectorError):
    """Erreur de validation de la configuration d'un connecteur."""


@dataclass
class IngestionResult:
    """
    Résultat standard retourné par tous les connecteurs.
    """

    source_id: str
    status: str
    files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def file_count(self) -> int:
        return len(self.files)


class BaseConnector(ABC):
    """
    Contrat commun à tous les connecteurs d'ingestion.
    """

    connector_type: str

    def __init__(
        self,
        source: SourceConfig,
        project_root: Path,
        ingestion_config: dict[str, Any],
    ) -> None:
        self.source = source
        self.project_root = project_root
        self.ingestion_config = ingestion_config

    @abstractmethod
    def validate(self) -> None:
        """
        Vérifie que la configuration de la source est valide.
        """

    @abstractmethod
    def ingest(self) -> IngestionResult:
        """
        Collecte ou découvre les fichiers de la source.
        """

    def resolve_path(self, relative_path: str | Path) -> Path:
        """
        Convertit un chemin relatif au projet en chemin absolu.
        """

        path = Path(relative_path)

        if path.is_absolute():
            return path.resolve()

        return (self.project_root / path).resolve()