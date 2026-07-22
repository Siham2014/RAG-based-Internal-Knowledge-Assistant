from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ChunkingStrategy(str, Enum):
    """
    Stratégies de chunking prises en charge par le benchmark.
    """

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class ChunkingModelError(ValueError):
    """
    Erreur déclenchée lorsqu'une configuration ou un chunk est invalide.
    """


@dataclass(slots=True)
class ChunkingConfig:
    """
    Configuration commune aux trois stratégies de chunking.

    Pour le SemanticChunker, chunk_size représente une taille cible.
    Pour FixedSizeChunker, chunk_size représente une limite exacte.
    Pour RecursiveChunker, chunk_size représente la taille maximale visée.
    """

    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int

    min_chunk_size: int = 1
    tokenizer_name: str = "default"
    preserve_sentences: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    include_heading_context: bool = True

    semantic_similarity_threshold: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            try:
                self.strategy = ChunkingStrategy(self.strategy)
            except ValueError as exc:
                supported = ", ".join(strategy.value for strategy in ChunkingStrategy)
                raise ChunkingModelError(
                    f"Stratégie inconnue : {self.strategy!r}. "
                    f"Stratégies disponibles : {supported}."
                ) from exc

        if self.chunk_size <= 0:
            raise ChunkingModelError(
                "chunk_size doit être strictement supérieur à zéro."
            )

        if self.chunk_overlap < 0:
            raise ChunkingModelError(
                "chunk_overlap ne peut pas être négatif."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingModelError(
                "chunk_overlap doit être strictement inférieur à chunk_size."
            )

        if self.min_chunk_size <= 0:
            raise ChunkingModelError(
                "min_chunk_size doit être strictement supérieur à zéro."
            )

        if self.min_chunk_size > self.chunk_size:
            raise ChunkingModelError(
                "min_chunk_size ne peut pas dépasser chunk_size."
            )

        if (
            self.semantic_similarity_threshold is not None
            and not 0.0 <= self.semantic_similarity_threshold <= 1.0
        ):
            raise ChunkingModelError(
                "semantic_similarity_threshold doit être compris entre 0 et 1."
            )

    @property
    def experiment_id(self) -> str:
        """
        Identifiant lisible de la configuration expérimentale.

        Exemple :
        recursive_512_64
        """

        return (
            f"{self.strategy.value}_"
            f"{self.chunk_size}_"
            f"{self.chunk_overlap}"
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strategy"] = self.strategy.value
        result["experiment_id"] = self.experiment_id
        return result
@dataclass(slots=True)
class TextBlock:
    """
    Représentation normalisée d'un bloc documentaire.

    Un bloc peut correspondre à :
    - un titre ;
    - un paragraphe ;
    - une liste ;
    - un tableau ;
    - un bloc de code ;
    - un autre contenu textuel.
    """

    block_id: str
    document_id: str
    block_type: str
    text: str
    block_index: int

    section_id: str | None = None
    section_heading: str | None = None
    heading_path: list[str] = field(default_factory=list)

    source_url: str | None = None
    document_title: str | None = None
    language: str | None = None

    start_position: int | None = None
    end_position: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block_id = self.block_id.strip()
        self.document_id = self.document_id.strip()
        self.block_type = self.block_type.strip().lower()
        self.text = self.text.strip()

        if not self.block_id:
            raise ChunkingModelError(
                "block_id ne peut pas être vide."
            )

        if not self.document_id:
            raise ChunkingModelError(
                "document_id ne peut pas être vide."
            )

        if not self.block_type:
            raise ChunkingModelError(
                "block_type ne peut pas être vide."
            )

        if not self.text:
            raise ChunkingModelError(
                "Le texte du bloc ne peut pas être vide."
            )

        if self.block_index < 0:
            raise ChunkingModelError(
                "block_index ne peut pas être négatif."
            )

        if (
            self.start_position is not None
            and self.end_position is not None
            and self.end_position < self.start_position
        ):
            raise ChunkingModelError(
                "end_position ne peut pas être inférieur à start_position."
            )

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        block_type: str,
        text: str,
        block_index: int,
        section_id: str | None = None,
        section_heading: str | None = None,
        heading_path: list[str] | None = None,
        source_url: str | None = None,
        document_title: str | None = None,
        language: str | None = None,
        start_position: int | None = None,
        end_position: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TextBlock":
        normalized_text = text.strip()

        identity = "|".join(
            [
                document_id.strip(),
                block_type.strip().lower(),
                str(block_index),
                normalized_text,
            ]
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return cls(
            block_id=f"block_{digest}",
            document_id=document_id,
            block_type=block_type,
            text=normalized_text,
            block_index=block_index,
            section_id=section_id,
            section_heading=section_heading,
            heading_path=list(heading_path or []),
            source_url=source_url,
            document_title=document_title,
            language=language,
            start_position=start_position,
            end_position=end_position,
            metadata=dict(metadata or {}),
        )

    @property
    def character_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentChunk:
    """
    Représentation normalisée d'un chunk documentaire.

    Le chunk conserve les informations nécessaires pour :
    - retrouver son document d'origine ;
    - générer un embedding ;
    - l'indexer dans PostgreSQL + pgvector ;
    - l'indexer dans le moteur lexical ;
    - afficher une citation dans la réponse finale.
    """

    chunk_id: str

    document_id: str
    text: str

    strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int

    chunk_index: int
    token_count: int
    character_count: int

    source_id: str | None = None
    source_url: str | None = None
    document_title: str | None = None

    section_id: str | None = None
    section_heading: str | None = None
    heading_path: list[str] = field(default_factory=list)

    content_types: list[str] = field(default_factory=list)

    start_position: int | None = None
    end_position: int | None = None

    source_checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            try:
                self.strategy = ChunkingStrategy(self.strategy)
            except ValueError as exc:
                raise ChunkingModelError(
                    f"Stratégie de chunk invalide : {self.strategy!r}."
                ) from exc

        self.document_id = self.document_id.strip()
        self.text = self.text.strip()

        if not self.chunk_id.strip():
            raise ChunkingModelError("chunk_id ne peut pas être vide.")

        if not self.document_id:
            raise ChunkingModelError("document_id ne peut pas être vide.")

        if not self.text:
            raise ChunkingModelError("Le texte du chunk ne peut pas être vide.")

        if self.chunk_index < 0:
            raise ChunkingModelError(
                "chunk_index ne peut pas être négatif."
            )

        if self.token_count <= 0:
            raise ChunkingModelError(
                "token_count doit être strictement supérieur à zéro."
            )

        if self.character_count != len(self.text):
            raise ChunkingModelError(
                "character_count doit correspondre à la longueur du texte."
            )

        if self.chunk_size <= 0:
            raise ChunkingModelError(
                "chunk_size doit être strictement supérieur à zéro."
            )

        if self.chunk_overlap < 0:
            raise ChunkingModelError(
                "chunk_overlap ne peut pas être négatif."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingModelError(
                "chunk_overlap doit être inférieur à chunk_size."
            )

        if (
            self.start_position is not None
            and self.end_position is not None
            and self.end_position < self.start_position
        ):
            raise ChunkingModelError(
                "end_position ne peut pas être inférieur à start_position."
            )

    @property
    def experiment_id(self) -> str:
        return (
            f"{self.strategy.value}_"
            f"{self.chunk_size}_"
            f"{self.chunk_overlap}"
        )

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        text: str,
        strategy: ChunkingStrategy | str,
        chunk_size: int,
        chunk_overlap: int,
        chunk_index: int,
        token_count: int,
        source_id: str | None = None,
        source_url: str | None = None,
        document_title: str | None = None,
        section_id: str | None = None,
        section_heading: str | None = None,
        heading_path: list[str] | None = None,
        content_types: list[str] | None = None,
        start_position: int | None = None,
        end_position: int | None = None,
        source_checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """
        Crée un DocumentChunk avec un identifiant déterministe.
        """

        normalized_text = text.strip()

        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)

        chunk_id = cls.generate_chunk_id(
            document_id=document_id,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index=chunk_index,
            text=normalized_text,
        )

        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            text=normalized_text,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_index=chunk_index,
            token_count=token_count,
            character_count=len(normalized_text),
            source_id=source_id,
            source_url=source_url,
            document_title=document_title,
            section_id=section_id,
            section_heading=section_heading,
            heading_path=list(heading_path or []),
            content_types=list(content_types or []),
            start_position=start_position,
            end_position=end_position,
            source_checksum=source_checksum,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def generate_chunk_id(
        *,
        document_id: str,
        strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
        chunk_index: int,
        text: str,
    ) -> str:
        """
        Génère un identifiant déterministe SHA-256.

        Le même document, la même configuration, le même index et le même
        texte produisent toujours le même chunk_id.
        """

        identity = "|".join(
            [
                document_id.strip(),
                strategy.value,
                str(chunk_size),
                str(chunk_overlap),
                str(chunk_index),
                text.strip(),
            ]
        )

        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        return f"chunk_{digest}"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strategy"] = self.strategy.value
        result["experiment_id"] = self.experiment_id
        return result