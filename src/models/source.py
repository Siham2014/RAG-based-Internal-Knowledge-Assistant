from dataclasses import dataclass, field
from typing import Any


class SourceConfigurationError(ValueError):
    """Erreur liée à la configuration d'une source."""


@dataclass(frozen=True)
class SourceConfig:
    """
    Représente une source documentaire déclarée dans sources.yaml.

    Le modèle reste générique afin de supporter différents types de sources :
    HTML, GitHub, dossier local, PDF distant, Notion, SharePoint, etc.
    """

    id: str
    name: str
    type: str
    enabled: bool = True

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceConfig":
        """Construit une source à partir d'un bloc YAML."""

        if not isinstance(data, dict):
            raise SourceConfigurationError(
                "La configuration d'une source doit être un dictionnaire."
            )

        required_fields = ["id", "name", "type"]

        missing_fields = [
            field_name
            for field_name in required_fields
            if not data.get(field_name)
        ]

        if missing_fields:
            raise SourceConfigurationError(
                "Champs obligatoires manquants : "
                + ", ".join(missing_fields)
            )

        return cls(
            id=str(data["id"]).strip(),
            name=str(data["name"]).strip(),
            type=str(data["type"]).strip().lower(),
            enabled=bool(data.get("enabled", True)),
            input=data.get("input", {}) or {},
            output=data.get("output", {}) or {},
            filters=data.get("filters", {}) or {},
            extraction=data.get("extraction", {}) or {},
            metadata=data.get("metadata", {}) or {},
        )