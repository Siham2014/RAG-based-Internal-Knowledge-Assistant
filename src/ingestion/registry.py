from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from src.ingestion.connectors.base import BaseConnector
from src.models.source import SourceConfig


ConnectorClass = TypeVar(
    "ConnectorClass",
    bound=type[BaseConnector],
)


class ConnectorRegistryError(Exception):
    """Erreur liée au registre des connecteurs."""


class ConnectorRegistry:
    """
    Registre central des connecteurs disponibles.

    Il associe un type défini dans sources.yaml à une classe de connecteur.
    """

    _connectors: dict[str, type[BaseConnector]] = {}

    @classmethod
    def register(
        cls,
        connector_type: str,
    ) -> Callable[[ConnectorClass], ConnectorClass]:
        """
        Décorateur permettant à un connecteur de s'enregistrer lui-même.

        Exemple :
            @ConnectorRegistry.register("html")
            class HtmlConnector(BaseConnector):
                ...
        """

        normalized_type = connector_type.strip().lower()

        if not normalized_type:
            raise ConnectorRegistryError(
                "Le type du connecteur ne peut pas être vide."
            )

        def decorator(
            connector_class: ConnectorClass,
        ) -> ConnectorClass:
            if normalized_type in cls._connectors:
                existing_class = cls._connectors[normalized_type]

                raise ConnectorRegistryError(
                    f"Le connecteur '{normalized_type}' est déjà enregistré "
                    f"par {existing_class.__name__}."
                )

            if not issubclass(connector_class, BaseConnector):
                raise ConnectorRegistryError(
                    f"{connector_class.__name__} doit hériter "
                    "de BaseConnector."
                )

            connector_class.connector_type = normalized_type
            cls._connectors[normalized_type] = connector_class

            return connector_class

        return decorator

    @classmethod
    def create(
        cls,
        source: SourceConfig,
        project_root: Path,
        ingestion_config: dict[str, Any],
    ) -> BaseConnector:
        """
        Instancie automatiquement le connecteur associé à la source.
        """

        connector_class = cls._connectors.get(source.type)

        if connector_class is None:
            available = ", ".join(sorted(cls._connectors)) or "aucun"

            raise ConnectorRegistryError(
                f"Aucun connecteur enregistré pour le type "
                f"'{source.type}'. Connecteurs disponibles : {available}"
            )

        return connector_class(
            source=source,
            project_root=project_root,
            ingestion_config=ingestion_config,
        )

    @classmethod
    def available_connectors(cls) -> tuple[str, ...]:
        """Retourne les connecteurs enregistrés."""

        return tuple(sorted(cls._connectors))