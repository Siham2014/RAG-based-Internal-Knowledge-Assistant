from __future__ import annotations

from src.chunking.models import (
    ChunkingConfig,
    ChunkingModelError,
    ChunkingStrategy,
    DocumentChunk,
)


def print_separator() -> None:
    print("=" * 70)


def test_chunking_configurations() -> None:
    print_separator()
    print("TEST DES CONFIGURATIONS DE CHUNKING")
    print_separator()

    configurations = []

    for strategy in ChunkingStrategy:
        for chunk_size, chunk_overlap in [
            (256, 32),
            (512, 64),
            (1024, 128),
        ]:
            config = ChunkingConfig(
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=1,
            )

            configurations.append(config)

            print(
                f"{config.experiment_id:<28} "
                f"strategy={config.strategy.value:<12} "
                f"size={config.chunk_size:<4} "
                f"overlap={config.chunk_overlap}"
            )

    assert len(configurations) == 9

    experiment_ids = {
        configuration.experiment_id
        for configuration in configurations
    }

    assert len(experiment_ids) == 9

    print()
    print("Nombre de configurations :", len(configurations))


def test_document_chunk() -> None:
    print()
    print_separator()
    print("TEST DU MODELE DOCUMENT CHUNK")
    print_separator()

    text = (
        "Azure availability zones are physically separate locations "
        "within an Azure region. They improve application resiliency."
    )

    chunk = DocumentChunk.create(
        document_id="azure-reliability-document",
        source_id="microsoft-learn",
        source_url="https://learn.microsoft.com/azure/reliability/",
        document_title="Azure Reliability",
        section_id="availability-zones",
        section_heading="Availability zones",
        heading_path=[
            "Azure Reliability",
            "Availability zones",
        ],
        content_types=[
            "heading",
            "paragraph",
        ],
        text=text,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=512,
        chunk_overlap=64,
        chunk_index=0,
        token_count=19,
        start_position=0,
        end_position=len(text),
        source_checksum="example-checksum",
        metadata={
            "language": "en",
            "organization": "Microsoft",
        },
    )

    print("Chunk ID :", chunk.chunk_id)
    print("Experiment ID :", chunk.experiment_id)
    print("Document ID :", chunk.document_id)
    print("Stratégie :", chunk.strategy.value)
    print("Chunk size :", chunk.chunk_size)
    print("Overlap :", chunk.chunk_overlap)
    print("Token count :", chunk.token_count)
    print("Character count :", chunk.character_count)
    print("Heading path :", chunk.heading_path)
    print("Texte :", chunk.text)

    assert chunk.chunk_id.startswith("chunk_")
    assert len(chunk.chunk_id) == len("chunk_") + 64
    assert chunk.document_id == "azure-reliability-document"
    assert chunk.strategy == ChunkingStrategy.RECURSIVE
    assert chunk.chunk_size == 512
    assert chunk.chunk_overlap == 64
    assert chunk.character_count == len(text)
    assert chunk.experiment_id == "recursive_512_64"

    duplicate = DocumentChunk.create(
        document_id="azure-reliability-document",
        source_id="microsoft-learn",
        source_url="https://learn.microsoft.com/azure/reliability/",
        document_title="Azure Reliability",
        section_id="availability-zones",
        section_heading="Availability zones",
        heading_path=[
            "Azure Reliability",
            "Availability zones",
        ],
        content_types=[
            "heading",
            "paragraph",
        ],
        text=text,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=512,
        chunk_overlap=64,
        chunk_index=0,
        token_count=19,
    )

    assert duplicate.chunk_id == chunk.chunk_id

    different_configuration = DocumentChunk.create(
        document_id="azure-reliability-document",
        text=text,
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=256,
        chunk_overlap=32,
        chunk_index=0,
        token_count=19,
    )

    assert different_configuration.chunk_id != chunk.chunk_id

    dictionary = chunk.to_dict()

    assert dictionary["strategy"] == "recursive"
    assert dictionary["experiment_id"] == "recursive_512_64"

    print()
    print("Identifiant déterministe : OK")
    print("Sérialisation en dictionnaire : OK")


def test_invalid_configuration() -> None:
    print()
    print_separator()
    print("TEST DES VALIDATIONS")
    print_separator()

    try:
        ChunkingConfig(
            strategy="fixed_size",
            chunk_size=256,
            chunk_overlap=256,
        )
    except ChunkingModelError as exc:
        print("Configuration invalide détectée :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour overlap >= chunk_size."
        )

    try:
        DocumentChunk.create(
            document_id="document-1",
            text="",
            strategy="fixed_size",
            chunk_size=256,
            chunk_overlap=32,
            chunk_index=0,
            token_count=1,
        )
    except ChunkingModelError as exc:
        print("Chunk invalide détecté :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour un texte vide."
        )


def main() -> None:
    test_chunking_configurations()
    test_document_chunk()
    test_invalid_configuration()

    print()
    print_separator()
    print("MODELES DE CHUNKING FONCTIONNELS.")
    print_separator()


if __name__ == "__main__":
    main()