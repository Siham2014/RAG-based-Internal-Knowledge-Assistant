from __future__ import annotations

from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    TextBlock,
)
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.registry import ChunkerRegistry
from src.chunking.tokenizer import WhitespaceTokenizer


def separator() -> None:
    print("=" * 70)


def create_blocks() -> list[TextBlock]:
    return [
        TextBlock.create(
            document_id="azure-recursive-test",
            block_type="heading",
            text="Azure reliability",
            block_index=0,
            section_id="overview",
            section_heading="Overview",
            heading_path=[
                "Azure reliability",
                "Overview",
            ],
            document_title="Azure reliability guide",
            source_url="https://example.com/reliability",
        ),
        TextBlock.create(
            document_id="azure-recursive-test",
            block_type="paragraph",
            text=(
                "Azure applications must remain available during "
                "infrastructure failures. Reliability requires "
                "redundancy, monitoring, recovery procedures and "
                "continuous validation. Applications should be "
                "designed to tolerate failures without losing "
                "critical business data."
            ),
            block_index=1,
            section_id="overview",
            section_heading="Overview",
            heading_path=[
                "Azure reliability",
                "Overview",
            ],
            document_title="Azure reliability guide",
            source_url="https://example.com/reliability",
        ),
        TextBlock.create(
            document_id="azure-recursive-test",
            block_type="heading",
            text="Availability zones",
            block_index=2,
            section_id="zones",
            section_heading="Availability zones",
            heading_path=[
                "Azure reliability",
                "Availability zones",
            ],
            document_title="Azure reliability guide",
            source_url="https://example.com/reliability",
        ),
        TextBlock.create(
            document_id="azure-recursive-test",
            block_type="paragraph",
            text=(
                "Availability zones are physically separate "
                "datacenter locations within an Azure region. "
                "They provide independent power, cooling and "
                "networking. Workloads can be distributed across "
                "multiple zones to reduce the impact of failures."
            ),
            block_index=3,
            section_id="zones",
            section_heading="Availability zones",
            heading_path=[
                "Azure reliability",
                "Availability zones",
            ],
            document_title="Azure reliability guide",
            source_url="https://example.com/reliability",
        ),
    ]


def create_config() -> ChunkingConfig:
    return ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=18,
        chunk_overlap=4,
        tokenizer_name="whitespace",
        preserve_sentences=True,
    )


def test_recursive_chunker() -> None:
    separator()
    print("TEST DU RECURSIVE CHUNKER")
    separator()

    tokenizer = WhitespaceTokenizer()

    chunker = RecursiveChunker(
        config=create_config(),
        tokenizer=tokenizer,
    )

    chunks = chunker.chunk_blocks(create_blocks())

    print("Nombre de chunks :", len(chunks))
    print("Chunk size :", chunker.config.chunk_size)
    print("Overlap :", chunker.config.chunk_overlap)
    print()

    assert chunks
    assert len(chunks) >= 3

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index} | "
            f"tokens={chunk.token_count}"
        )
        print("Chunk ID :", chunk.chunk_id)
        print("Types :", chunk.content_types)
        print("Sections :", chunk.metadata["section_ids"])
        print("Heading path :", chunk.heading_path)
        print(
            "Overlap déclaré :",
            chunk.metadata["overlap_token_count"],
        )
        print("Texte :", chunk.text)
        print("-" * 70)

        assert chunk.strategy == ChunkingStrategy.RECURSIVE
        assert chunk.token_count <= 18
        assert chunk.token_count == tokenizer.count_tokens(
            chunk.text
        )
        assert chunk.document_id == "azure-recursive-test"

    print()
    print("Découpage récursif validé : OK")


def test_sentence_preservation() -> None:
    print()
    separator()
    print("TEST DE PRESERVATION DES PHRASES")
    separator()

    block = TextBlock.create(
        document_id="sentence-test",
        block_type="paragraph",
        text=(
            "First sentence contains useful information. "
            "Second sentence describes reliability principles. "
            "Third sentence explains failure recovery."
        ),
        block_index=0,
    )

    tokenizer = WhitespaceTokenizer()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=7,
        chunk_overlap=0,
        tokenizer_name="whitespace",
        preserve_sentences=True,
    )

    chunker = RecursiveChunker(
        config=config,
        tokenizer=tokenizer,
    )

    chunks = chunker.chunk_blocks([block])

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index}: "
            f"{chunk.text}"
        )
        assert chunk.token_count <= 7

    assert len(chunks) >= 2

    print("Préservation des phrases : OK")


def test_oversized_block_fallback() -> None:
    print()
    separator()
    print("TEST DU FALLBACK PAR TOKENS")
    separator()

    text = " ".join(
        f"token_{index}"
        for index in range(25)
    )

    block = TextBlock.create(
        document_id="fallback-test",
        block_type="code",
        text=text,
        block_index=0,
    )

    tokenizer = WhitespaceTokenizer()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE,
        chunk_size=10,
        chunk_overlap=2,
        tokenizer_name="whitespace",
        preserve_code_blocks=True,
    )

    chunker = RecursiveChunker(
        config=config,
        tokenizer=tokenizer,
    )

    chunks = chunker.chunk_blocks([block])

    print("Nombre de chunks :", len(chunks))

    for chunk in chunks:
        print(
            chunk.chunk_index,
            chunk.token_count,
            chunk.text,
        )

        assert chunk.token_count <= 10
        assert "code" in chunk.content_types

    assert len(chunks) >= 3

    print("Fallback par tokens : OK")


def test_complete_document() -> None:
    print()
    separator()
    print("TEST AVEC UN DOCUMENT COMPLET")
    separator()

    document = {
        "document_id": "complete-document",
        "title": "Azure resilience",
        "source_url": "https://example.com/resilience",
        "sections": [
            {
                "section_id": "design",
                "heading": "Design for failure",
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Applications should anticipate "
                            "infrastructure failures. Redundancy "
                            "and monitoring improve resilience."
                        ),
                    },
                    {
                        "type": "list",
                        "text": (
                            "Use multiple zones. Monitor health. "
                            "Test recovery procedures."
                        ),
                    },
                ],
            }
        ],
    }

    chunker = RecursiveChunker(
        config=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=12,
            chunk_overlap=2,
            tokenizer_name="whitespace",
        ),
        tokenizer=WhitespaceTokenizer(),
    )

    chunks = chunker.chunk(document)

    assert chunks
    assert all(
        chunk.document_id == "complete-document"
        for chunk in chunks
    )

    print("Nombre de chunks :", len(chunks))
    print("Document complet : OK")


def test_registry_creation() -> None:
    print()
    separator()
    print("TEST AVEC CHUNKER REGISTRY")
    separator()

    ChunkerRegistry.register(
        ChunkingStrategy.RECURSIVE,
        RecursiveChunker,
        replace=True,
    )

    chunker = ChunkerRegistry.create(create_config())

    assert isinstance(chunker, RecursiveChunker)

    chunks = chunker.chunk_blocks(create_blocks())

    assert chunks

    print(
        "Stratégies enregistrées :",
        ChunkerRegistry.available_strategies(),
    )
    print("Chunker créé :", chunker)
    print("Création depuis le registre : OK")


def test_deterministic_ids() -> None:
    print()
    separator()
    print("TEST DES IDENTIFIANTS DETERMINISTES")
    separator()

    config = create_config()

    first_chunker = RecursiveChunker(
        config=config,
        tokenizer=WhitespaceTokenizer(),
    )

    second_chunker = RecursiveChunker(
        config=config,
        tokenizer=WhitespaceTokenizer(),
    )

    first_chunks = first_chunker.chunk_blocks(create_blocks())
    second_chunks = second_chunker.chunk_blocks(create_blocks())

    first_ids = [chunk.chunk_id for chunk in first_chunks]
    second_ids = [chunk.chunk_id for chunk in second_chunks]

    assert first_ids == second_ids

    print("Identifiants déterministes : OK")


def test_overlap_metadata() -> None:
    print()
    separator()
    print("TEST DES METADONNEES D'OVERLAP")
    separator()

    tokenizer = WhitespaceTokenizer()
    config = create_config()

    chunker = RecursiveChunker(
        config=config,
        tokenizer=tokenizer,
    )

    chunks = chunker.chunk_blocks(create_blocks())

    overlaps_detected = 0

    for index, chunk in enumerate(chunks):
        overlap_count = chunk.metadata[
            "overlap_token_count"
        ]

        print(
            f"Chunk {index} : "
            f"overlap_token_count={overlap_count}"
        )

        assert 0 <= overlap_count <= config.chunk_overlap

        if index == 0:
            assert overlap_count == 0
            continue

        if overlap_count > 0:
            overlaps_detected += 1

            previous_suffix = tokenizer.take_last_tokens(
                chunks[index - 1].text,
                overlap_count,
            )

            current_prefix = tokenizer.take_first_tokens(
                chunk.text,
                overlap_count,
            )

            print(
                "Fin du chunk précédent :",
                previous_suffix,
            )
            print(
                "Début du chunk courant :",
                current_prefix,
            )

            assert previous_suffix == current_prefix

    assert overlaps_detected > 0

    print("Métadonnées d'overlap : OK")


def main() -> None:
    test_recursive_chunker()
    test_sentence_preservation()
    test_oversized_block_fallback()
    test_complete_document()
    test_registry_creation()
    test_deterministic_ids()
    test_overlap_metadata()

    print()
    separator()
    print("RECURSIVE CHUNKER FONCTIONNEL.")
    separator()


if __name__ == "__main__":
    main()