from __future__ import annotations

from src.chunking.fixed_size_chunker import (
    FixedSizeChunker,
)
from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    TextBlock,
)
from src.chunking.registry import ChunkerRegistry
from src.chunking.tokenizer import WhitespaceTokenizer


def print_separator() -> None:
    print("=" * 70)


def build_blocks() -> list[TextBlock]:
    return [
        TextBlock.create(
            document_id="azure-document",
            block_type="heading",
            text="Azure reliability overview",
            block_index=0,
            section_id="overview",
            section_heading="Overview",
            heading_path=[
                "Azure Reliability",
                "Overview",
            ],
            document_title="Azure Reliability",
            source_url=(
                "https://learn.microsoft.com/"
                "azure/reliability/"
            ),
        ),
        TextBlock.create(
            document_id="azure-document",
            block_type="paragraph",
            text=(
                "Azure applications must remain available "
                "during infrastructure failures"
            ),
            block_index=1,
            section_id="overview",
            section_heading="Overview",
            heading_path=[
                "Azure Reliability",
                "Overview",
            ],
            document_title="Azure Reliability",
            source_url=(
                "https://learn.microsoft.com/"
                "azure/reliability/"
            ),
        ),
        TextBlock.create(
            document_id="azure-document",
            block_type="paragraph",
            text=(
                "Availability zones provide physically "
                "separate datacenter locations"
            ),
            block_index=2,
            section_id="availability-zones",
            section_heading="Availability zones",
            heading_path=[
                "Azure Reliability",
                "Availability zones",
            ],
            document_title="Azure Reliability",
            source_url=(
                "https://learn.microsoft.com/"
                "azure/reliability/"
            ),
        ),
        TextBlock.create(
            document_id="azure-document",
            block_type="list",
            text=(
                "Use redundancy monitor health "
                "and design for failure"
            ),
            block_index=3,
            section_id="recommendations",
            section_heading="Recommendations",
            heading_path=[
                "Azure Reliability",
                "Recommendations",
            ],
            document_title="Azure Reliability",
            source_url=(
                "https://learn.microsoft.com/"
                "azure/reliability/"
            ),
        ),
    ]


def test_fixed_size_chunking() -> None:
    print_separator()
    print("TEST DU FIXED SIZE CHUNKER")
    print_separator()

    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=8,
        chunk_overlap=2,
        tokenizer_name="whitespace",
    )

    chunker = FixedSizeChunker(
        config=config,
        tokenizer=WhitespaceTokenizer(),
    )

    blocks = build_blocks()
    chunks = chunker.chunk(blocks)

    total_tokens = sum(
        len(block.text.split())
        for block in blocks
    )

    print("Nombre de tokens du document :", total_tokens)
    print("Chunk size :", config.chunk_size)
    print("Overlap :", config.chunk_overlap)
    print("Step :", config.chunk_size - config.chunk_overlap)
    print("Nombre de chunks :", len(chunks))
    print()

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index} | "
            f"tokens={chunk.token_count} | "
            f"types={chunk.content_types}"
        )
        print("Chunk ID :", chunk.chunk_id)
        print(
            "Position tokens :",
            chunk.metadata["start_token"],
            "→",
            chunk.metadata["end_token"],
        )
        print(
            "Blocs couverts :",
            chunk.metadata["covered_block_indexes"],
        )
        print(
            "Sections :",
            chunk.metadata["section_headings"],
        )
        print("Heading path :", chunk.heading_path)
        print("Texte :", chunk.text)
        print("-" * 70)

    assert total_tokens == 26
    assert len(chunks) == 4

    assert chunks[0].token_count == 8
    assert chunks[0].metadata["start_token"] == 0
    assert chunks[0].metadata["end_token"] == 8

    assert chunks[1].token_count == 8
    assert chunks[1].metadata["start_token"] == 6
    assert chunks[1].metadata["end_token"] == 14

    assert chunks[2].metadata["start_token"] == 12
    assert chunks[2].metadata["end_token"] == 20

    assert chunks[3].metadata["start_token"] == 18
    assert chunks[3].metadata["end_token"] == 26
    assert chunks[3].token_count == 8

    for chunk in chunks:
        assert chunk.strategy == ChunkingStrategy.FIXED_SIZE
        assert chunk.chunk_size == 8
        assert chunk.chunk_overlap == 2
        assert chunk.experiment_id == "fixed_size_8_2"
        assert chunk.token_count <= 8
        assert chunk.chunk_id.startswith("chunk_")

    print()
    print("Découpage fixe validé : OK")


def test_overlap_content() -> None:
    print()
    print_separator()
    print("TEST DU CONTENU DE L'OVERLAP")
    print_separator()

    config = ChunkingConfig(
        strategy="fixed_size",
        chunk_size=8,
        chunk_overlap=2,
        tokenizer_name="whitespace",
    )

    tokenizer = WhitespaceTokenizer()

    chunker = FixedSizeChunker(
        config=config,
        tokenizer=tokenizer,
    )

    chunks = chunker.chunk(build_blocks())

    first_ids = tokenizer.encode(
        chunks[0].text
    )

    second_ids = tokenizer.encode(
        chunks[1].text
    )

    assert first_ids[-2:] == second_ids[:2]

    second_ids = tokenizer.encode(
        chunks[1].text
    )

    third_ids = tokenizer.encode(
        chunks[2].text
    )

    assert second_ids[-2:] == third_ids[:2]

    print("Overlap réel entre les chunks : OK")


def test_document_input() -> None:
    print()
    print_separator()
    print("TEST AVEC UN DOCUMENT COMPLET")
    print_separator()

    document = {
        "document_id": "document-from-builder",
        "title": "Azure document",
        "source_url": "https://learn.microsoft.com/",
        "sections": [
            {
                "id": "section-1",
                "type": "heading",
                "heading": "Reliability",
                "text": "Reliability",
                "children": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Azure services provide resilient "
                            "cloud infrastructure for applications"
                        ),
                    },
                ],
            },
        ],
    }

    config = ChunkingConfig(
        strategy="fixed_size",
        chunk_size=6,
        chunk_overlap=1,
        tokenizer_name="whitespace",
    )

    chunker = FixedSizeChunker(config)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].document_id == (
        "document-from-builder"
    )

    print("Document transformé et découpé : OK")
    print("Nombre de chunks :", len(chunks))


def test_registry_creation() -> None:
    print()
    print_separator()
    print("TEST AVEC CHUNKER REGISTRY")
    print_separator()

    ChunkerRegistry.clear()

    ChunkerRegistry.register(
        ChunkingStrategy.FIXED_SIZE,
        FixedSizeChunker,
    )

    config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=8,
        chunk_overlap=2,
        tokenizer_name="whitespace",
    )

    chunker = ChunkerRegistry.create(config)

    chunks = chunker.chunk(build_blocks())

    assert isinstance(chunker, FixedSizeChunker)
    assert len(chunks) == 4

    print("Création depuis le registre : OK")
    print("Chunker :", chunker)


def test_deterministic_chunks() -> None:
    print()
    print_separator()
    print("TEST DES CHUNKS DETERMINISTES")
    print_separator()

    config = ChunkingConfig(
        strategy="fixed_size",
        chunk_size=8,
        chunk_overlap=2,
        tokenizer_name="whitespace",
    )

    first_chunker = FixedSizeChunker(config)
    second_chunker = FixedSizeChunker(config)

    first_result = first_chunker.chunk(
        build_blocks()
    )

    second_result = second_chunker.chunk(
        build_blocks()
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first_result
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_result
    ]

    assert first_ids == second_ids

    print("Identifiants déterministes : OK")


def main() -> None:
    test_fixed_size_chunking()
    test_overlap_content()
    test_document_input()
    test_registry_creation()
    test_deterministic_chunks()

    print()
    print_separator()
    print("FIXED SIZE CHUNKER FONCTIONNEL.")
    print_separator()


if __name__ == "__main__":
    main()