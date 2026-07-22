from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(Exception):
    """Erreur liée au chargement ou à la validation de la configuration."""


class ConfigLoader:
    """Charge les fichiers YAML du projet."""

    def __init__(self, project_root: Path | None = None) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = project_root

    def load_yaml(self, relative_path: str | Path) -> dict[str, Any]:
        """
        Charge un fichier YAML à partir de la racine du projet.
        """

        config_path = self.project_root / relative_path

        if not config_path.exists():
            raise ConfigurationError(
                f"Fichier de configuration introuvable : {config_path}"
            )

        if not config_path.is_file():
            raise ConfigurationError(
                f"Le chemin n'est pas un fichier : {config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

        except yaml.YAMLError as error:
            raise ConfigurationError(
                f"Le fichier YAML est invalide : {config_path}\n{error}"
            ) from error

        if data is None:
            raise ConfigurationError(
                f"Le fichier YAML est vide : {config_path}"
            )

        if not isinstance(data, dict):
            raise ConfigurationError(
                f"La racine du fichier YAML doit être un dictionnaire : "
                f"{config_path}"
            )

        return data

    def load_sources(self) -> dict[str, Any]:
        """Charge la configuration des sources."""

        config = self.load_yaml("config/sources.yaml")

        if "sources" not in config:
            raise ConfigurationError(
                "La clé 'sources' est absente de config/sources.yaml"
            )

        if not isinstance(config["sources"], list):
            raise ConfigurationError(
                "La clé 'sources' doit contenir une liste."
            )

        return config

    def load_ingestion(self) -> dict[str, Any]:
        """Charge la configuration générale de l'ingestion."""

        return self.load_yaml("config/ingestion.yaml")