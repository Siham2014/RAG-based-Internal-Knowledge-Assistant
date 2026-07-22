from __future__ import annotations

from typing import Any

from src.chunking.base import BaseChunker, ChunkingError
from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentChunk,
)
from src.chunking.registry import (
    ChunkerRegistry,
    ChunkerRegistryError,
)


class FakeFixedSizeChunker(BaseChunker):
    strategy_name = "fixed_size"

    def chunk(self, document: Any) -> list[DocumentChunk]:
        self.validate_document(document)

        text = str(document).strip()

        if not text:
            raise ChunkingError(
                "Le document ne contient aucun texte."
            )

        return [
            DocumentChunk.create(
                document_id="test-document",
                text=text,
                strategy=self.config.strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                chunk_index=0,
                token_count=len(text.split()),
            )
        ]


class FakeRecursiveChunker(BaseChunker):
    strategy_name = "recursive"

    def chunk(self, document: Any) -> list[DocumentChunk]:
        self.validate_document(document)

        text = str(document).strip()

        return [
            DocumentChunk.create(
                document_id="recursive-document",
                text=text,
                strategy=self.config.strategy,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                chunk_index=0,
                token_count=max(1, len(text.split())),
            )
        ]


def print_separator() -> None:
    print("=" * 70)


def test_registration() -> None:
    print_separator()
    print("TEST DE L'ENREGISTREMENT DES CHUNKERS")
    print_separator()

    ChunkerRegistry.clear()

    ChunkerRegistry.register(
        ChunkingStrategy.FIXED_SIZE,
        FakeFixedSizeChunker,
    )

    ChunkerRegistry.register(
        ChunkingStrategy.RECURSIVE,
        FakeRecursiveChunker,
    )

    available = ChunkerRegistry.available_strategies()

    print("Stratégies enregistrées :", available)

    assert available == [
        "fixed_size",
        "recursive",
    ]

    assert ChunkerRegistry.is_registered("fixed_size")
    assert ChunkerRegistry.is_registered("recursive")
    assert not ChunkerRegistry.is_registered("semantic")


def test_chunker_creation() -> None:
    print()
    print_separator()
    print("TEST DE CREATION D'UN CHUNKER")
    print_separator()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=256,
        chunk_overlap=32,
    )

    chunker = ChunkerRegistry.create(config)

    print("Chunker créé :", chunker)
    print("Experiment ID :", chunker.experiment_id)
    print("Configuration :", chunker.get_configuration())

    assert isinstance(chunker, FakeFixedSizeChunker)
    assert chunker.experiment_id == "fixed_size_256_32"

    chunks = chunker.chunk(
        "Azure reliability documentation."
    )

    assert len(chunks) == 1
    assert chunks[0].strategy == ChunkingStrategy.FIXED_SIZE
    assert chunks[0].experiment_id == "fixed_size_256_32"

    print("Nombre de chunks :", len(chunks))
    print("Chunk ID :", chunks[0].chunk_id)
    print("Texte :", chunks[0].text)


def test_duplicate_registration() -> None:
    print()
    print_separator()
    print("TEST D'UN DOUBLON D'ENREGISTREMENT")
    print_separator()

    try:
        ChunkerRegistry.register(
            ChunkingStrategy.FIXED_SIZE,
            FakeFixedSizeChunker,
        )
    except ChunkerRegistryError as exc:
        print("Doublon détecté :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour un doublon."
        )


def test_unregistered_strategy() -> None:
    print()
    print_separator()
    print("TEST D'UNE STRATEGIE NON ENREGISTREE")
    print_separator()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=512,
        chunk_overlap=64,
    )

    try:
        ChunkerRegistry.create(config)
    except ChunkerRegistryError as exc:
        print("Stratégie absente détectée :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour semantic."
        )


def test_invalid_chunker_class() -> None:
    print()
    print_separator()
    print("TEST D'UNE CLASSE INVALIDE")
    print_separator()

    class InvalidChunker:
        pass

    try:
        ChunkerRegistry.register(
            ChunkingStrategy.SEMANTIC,
            InvalidChunker,
        )
    except ChunkerRegistryError as exc:
        print("Classe invalide détectée :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour InvalidChunker."
        )


def test_none_document() -> None:
    print()
    print_separator()
    print("TEST D'UN DOCUMENT NONE")
    print_separator()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=256,
        chunk_overlap=32,
    )

    chunker = ChunkerRegistry.create(config)

    try:
        chunker.chunk(None)
    except ChunkingError as exc:
        print("Document invalide détecté :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour document=None."
        )


def main() -> None:
    test_registration()
    test_chunker_creation()
    test_duplicate_registration()
    test_unregistered_strategy()
    test_invalid_chunker_class()
    test_none_document()

    print()
    print_separator()
    print("BASECHUNKER ET CHUNKERREGISTRY FONCTIONNELS.")
    print_separator()


if __name__ == "__main__":
    main()