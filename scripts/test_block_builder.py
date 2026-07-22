from __future__ import annotations

from src.chunking.block_builder import (
    BlockBuilder,
    BlockBuilderError,
)


def print_separator() -> None:
    print("=" * 70)


def build_test_document() -> dict:
    return {
        "document_id": "azure-reliability",
        "title": "Azure Reliability",
        "source_url": (
            "https://learn.microsoft.com/azure/"
            "reliability/"
        ),
        "language": "en",
        "sections": [
            {
                "id": "overview",
                "type": "heading",
                "heading": "Overview",
                "text": "Overview",
                "children": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Reliability ensures that an application "
                            "can meet its availability commitments."
                        ),
                    },
                    {
                        "type": "list",
                        "text": [
                            "Design for failure.",
                            "Use redundancy.",
                            "Monitor application health.",
                        ],
                    },
                ],
            },
            {
                "id": "availability-zones",
                "type": "heading",
                "heading": "Availability zones",
                "text": "Availability zones",
                "children": [
                    {
                        "type": "paragraph",
                        "text": (
                            "Availability zones are physically separate "
                            "locations within an Azure region."
                        ),
                    },
                    {
                        "type": "table",
                        "text": (
                            "Service | Zone support\n"
                            "Virtual Machines | Yes\n"
                            "Storage | Yes"
                        ),
                    },
                    {
                        "type": "code_block",
                        "text": (
                            "az vm create "
                            "--resource-group example"
                        ),
                    },
                ],
            },
        ],
    }


def test_block_creation() -> None:
    print_separator()
    print("TEST DE CREATION DES TEXTBLOCKS")
    print_separator()

    builder = BlockBuilder()
    document = build_test_document()

    blocks = builder.build(document)

    print("Nombre de blocs :", len(blocks))
    print()

    for block in blocks:
        print(
            f"Index={block.block_index} | "
            f"Type={block.block_type:<10} | "
            f"Section={block.section_heading}"
        )
        print("Block ID :", block.block_id)
        print("Heading path :", block.heading_path)
        print("Texte :", block.text)
        print("-" * 70)

    assert len(blocks) == 7

    assert blocks[0].block_type == "heading"
    assert blocks[0].text == "Overview"

    assert blocks[1].block_type == "paragraph"

    assert blocks[2].block_type == "list"
    assert "Design for failure." in blocks[2].text

    assert blocks[3].block_type == "heading"
    assert blocks[3].text == "Availability zones"

    assert blocks[4].block_type == "paragraph"
    assert blocks[5].block_type == "table"
    assert blocks[6].block_type == "code"

    assert blocks[6].heading_path == [
        "Availability zones"
    ]

    for index, block in enumerate(blocks):
        assert block.block_index == index
        assert block.block_id.startswith("block_")
        assert len(block.block_id) == len("block_") + 64

    print()
    print("Création des blocs : OK")


def test_deterministic_ids() -> None:
    print()
    print_separator()
    print("TEST DES IDENTIFIANTS DETERMINISTES")
    print_separator()

    builder = BlockBuilder()
    document = build_test_document()

    first_result = builder.build(document)
    second_result = builder.build(document)

    first_ids = [
        block.block_id
        for block in first_result
    ]

    second_ids = [
        block.block_id
        for block in second_result
    ]

    assert first_ids == second_ids

    print("Identifiants déterministes : OK")


def test_simple_text_document() -> None:
    print()
    print_separator()
    print("TEST D'UN DOCUMENT TEXTUEL SIMPLE")
    print_separator()

    builder = BlockBuilder()

    document = {
        "document_id": "simple-document",
        "title": "Simple document",
        "text": (
            "This document contains a single paragraph."
        ),
    }

    blocks = builder.build(document)

    assert len(blocks) == 1
    assert blocks[0].block_type == "paragraph"
    assert blocks[0].text == (
        "This document contains a single paragraph."
    )

    print("Document simple transformé : OK")


def test_empty_document() -> None:
    print()
    print_separator()
    print("TEST D'UN DOCUMENT VIDE")
    print_separator()

    builder = BlockBuilder()

    try:
        builder.build(
            {
                "document_id": "empty-document",
                "sections": [],
            }
        )
    except BlockBuilderError as exc:
        print("Document vide détecté :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour un document vide."
        )


def test_missing_document_id() -> None:
    print()
    print_separator()
    print("TEST D'UN IDENTIFIANT ABSENT")
    print_separator()

    builder = BlockBuilder()

    try:
        builder.build(
            {
                "title": "Document without identifier",
                "text": "Some content.",
            }
        )
    except BlockBuilderError as exc:
        print("Identifiant absent détecté :", exc)
    else:
        raise AssertionError(
            "Une erreur était attendue pour document_id absent."
        )


def main() -> None:
    test_block_creation()
    test_deterministic_ids()
    test_simple_text_document()
    test_empty_document()
    test_missing_document_id()

    print()
    print_separator()
    print("BLOCKBUILDER FONCTIONNEL.")
    print_separator()


if __name__ == "__main__":
    main()