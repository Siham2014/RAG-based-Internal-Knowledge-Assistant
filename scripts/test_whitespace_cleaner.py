import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import (
    CleanerRegistry,
    WhitespaceCleaner,
)
from src.models.document import (
    CodeBlock,
    DocumentLink,
    DocumentSection,
    DocumentTable,
    ParsedDocument,
)


def create_test_document() -> ParsedDocument:
    return ParsedDocument(
        document_id="whitespace_test:document",
        source_id="whitespace_test",
        source_type="local_folder",
        local_path="data/test.md",
        checksum_sha256="test-checksum",
        title="   Azure     Reliability   ",
        author="  Microsoft  ",
        source_url=(
            "  https://learn.microsoft.com/azure/  "
        ),
        language="  en  ",
        organization="  Microsoft  ",
        document_format="markdown",
        sections=[
            DocumentSection(
                section_id=(
                    "whitespace_test:"
                    "document:section-0"
                ),
                heading=(
                    "   Reliability     principles   "
                ),
                heading_level=1,
                position=0,
                heading_path=[
                    "  Azure   ",
                    "  Reliability     principles ",
                    "   ",
                ],
                paragraphs=[
                    (
                        "  Azure     workloads must "
                        "remain available.   "
                    ),
                    (
                        "First line.   \r\n"
                        "Second line.  \r\n\r\n\r\n"
                        "Third paragraph. "
                    ),
                    "      ",
                ],
                lists=[
                    [
                        "   Use    availability zones ",
                        " Configure    monitoring ",
                        "   ",
                    ],
                    [],
                ],
                tables=[
                    DocumentTable(
                        headers=[
                            "  Service ",
                            "   SLA   ",
                        ],
                        rows=[
                            [
                                " Virtual   Machines ",
                                " 99.99% ",
                            ],
                            [
                                "   ",
                                "   ",
                            ],
                        ],
                    )
                ],
                code_blocks=[
                    CodeBlock(
                        content=(
                            "\n\n"
                            "def health_check():    \n"
                            "    return 'healthy'    \n"
                            "\n"
                        ),
                        language="  python  ",
                    ),
                    CodeBlock(
                        content="   \n\n",
                        language=None,
                    ),
                ],
                links=[
                    DocumentLink(
                        text=(
                            "  Azure    documentation "
                        ),
                        url=(
                            "  https://learn.microsoft.com/  "
                        ),
                    ),
                    DocumentLink(
                        text=" Empty link ",
                        url="    ",
                    ),
                ],
            )
        ],
        metadata={
            "domain": "Azure",
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU WHITESPACE CLEANER")
    print("=" * 70)

    document = create_test_document()

    cleaner = CleanerRegistry.create(
        "whitespace"
    )

    assert isinstance(
        cleaner,
        WhitespaceCleaner,
    )

    cleaned_document = cleaner.run(
        document
    )

    section = cleaned_document.sections[0]

    print(
        "\nTitre original :",
        repr(document.title),
    )

    print(
        "Titre nettoyé :",
        repr(cleaned_document.title),
    )

    print(
        "\nTitre de section :",
        repr(section.heading),
    )

    print(
        "Chemin :",
        section.heading_path,
    )

    print(
        "Paragraphes :",
        section.paragraphs,
    )

    print(
        "Listes :",
        section.lists,
    )

    print(
        "Tableaux :",
        [
            {
                "headers": table.headers,
                "rows": table.rows,
            }
            for table in section.tables
        ],
    )

    print(
        "Code :",
        [
            {
                "language": code.language,
                "content": code.content,
            }
            for code in section.code_blocks
        ],
    )

    print(
        "Liens :",
        [
            {
                "text": link.text,
                "url": link.url,
            }
            for link in section.links
        ],
    )

    print(
        "Cleaners appliqués :",
        cleaned_document.metadata.get(
            "applied_cleaners"
        ),
    )

    assert document.title == (
        "   Azure     Reliability   "
    )

    assert cleaned_document.title == (
        "Azure Reliability"
    )

    assert cleaned_document.author == (
        "Microsoft"
    )

    assert cleaned_document.language == "en"

    assert cleaned_document.organization == (
        "Microsoft"
    )

    assert section.heading == (
        "Reliability principles"
    )

    assert section.heading_path == [
        "Azure",
        "Reliability principles",
    ]

    assert section.paragraphs[0] == (
        "Azure workloads must remain available."
    )

    assert section.paragraphs[1] == (
        "First line.\n"
        "Second line.\n\n"
        "Third paragraph."
    )

    assert len(
        section.paragraphs
    ) == 2

    assert section.lists == [
        [
            "Use availability zones",
            "Configure monitoring",
        ]
    ]

    assert (
        section.tables[0].headers
        == ["Service", "SLA"]
    )

    assert (
        section.tables[0].rows
        == [
            [
                "Virtual Machines",
                "99.99%",
            ]
        ]
    )

    assert len(
        section.code_blocks
    ) == 1

    assert (
        section.code_blocks[0].language
        == "python"
    )

    assert (
        section.code_blocks[0].content
        == (
            "def health_check():\n"
            "    return 'healthy'"
        )
    )

    assert len(
        section.links
    ) == 1

    assert (
        section.links[0].text
        == "Azure documentation"
    )

    assert (
        section.links[0].url
        == "https://learn.microsoft.com/"
    )

    assert (
        "whitespace_cleaner"
        in cleaned_document.metadata[
            "applied_cleaners"
        ]
    )

    print(
        "\nTexte nettoyé :\n"
    )

    print(cleaned_document.text)

    print(
        "\n WhitespaceCleaner fonctionnel."
    )


if __name__ == "__main__":
    main()