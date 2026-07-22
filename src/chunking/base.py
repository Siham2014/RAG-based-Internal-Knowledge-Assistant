from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.chunking.models import ChunkingConfig, DocumentChunk


class ChunkingError(RuntimeError):
    """
    Erreur générale déclenchée pendant une opération de chunking.
    """


class BaseChunker(ABC):
    """
    Contrat commun à toutes les stratégies de chunking.

    Toutes les stratégies doivent :
    - recevoir une configuration valide ;
    - transformer un document en liste de DocumentChunk ;
    - exposer leur nom ;
    - pouvoir être enregistrées dans ChunkerRegistry.
    """

    strategy_name: str

    def __init__(self, config: ChunkingConfig) -> None:
        if not isinstance(config, ChunkingConfig):
            raise TypeError(
                "config doit être une instance de ChunkingConfig."
            )

        self.config = config

    @abstractmethod
    def chunk(self, document: Any) -> list[DocumentChunk]:
        """
        Découpe un document en chunks.

        Le type exact du document sera connecté plus tard à ParsedDocument.
        Pour le moment, Any évite de coupler prématurément le module chunking
        avec le module parsing.
        """

        raise NotImplementedError

    def validate_document(self, document: Any) -> None:
        """
        Vérifie qu'un document a bien été fourni.
        """

        if document is None:
            raise ChunkingError(
                "Le document à découper ne peut pas être None."
            )

    @property
    def experiment_id(self) -> str:
        """
        Retourne l'identifiant de l'expérience liée au chunker.
        """

        return self.config.experiment_id

    def get_configuration(self) -> dict[str, Any]:
        """
        Retourne la configuration sérialisée du chunker.
        """

        return self.config.to_dict()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.config.strategy.value!r}, "
            f"chunk_size={self.config.chunk_size}, "
            f"chunk_overlap={self.config.chunk_overlap}"
            f")"
        )