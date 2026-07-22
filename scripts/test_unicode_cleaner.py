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
    UnicodeCleaner,
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
        document_id="unicode_test:document",
        source_id="unicode_test",
        source_type="local_folder",
        local_path="data/\ufeffazure\u00a0guide.md",
        checksum_sha256="unicode-test",
        title="\ufeffAzure\u00a0Reliability\u200b",
        author="Microsoft\u202fCorporation",
        source_url=(
            "https://learn.microsoft.com/"
            "azure\u200b/reliability"
        ),
        language="en",
        organization="Microsoft",
        document_format="markdown",
        sections=[
            DocumentSection(
                section_id=(
                    "unicode_test:"
                    "document:section-0"
                ),
                heading=(
                    "Reliability\u2014principles"
                ),
                heading_level=1,
                position=0,
                heading_path=[
                    "Azure\u00a0Architecture",
                    "Reliability\u2013principles",
                ],
                paragraphs=[
                    (
                        "\u201cAzure\u201d workloads "
                        "must remain available\u2026"
                    ),
                    (
                        "Use availability\u00a0zones "
                        "and geo\u2011replication."
                    ),
                    (
                        "French accents remain: "
                        "fiabilité, sécurité, résilience."
                    ),
                ],
                lists=[
                    [
                        "Use\u200b monitoring",
                        "Protect\u00a0data",
                    ]
                ],
                tables=[
                    DocumentTable(
                        headers=[
                            "Service",
                            "SLA\u202fvalue",
                        ],
                        rows=[
                            [
                                "Virtual\u00a0Machines",
                                "99.99\u202f%",
                            ]
                        ],
                    )
                ],
                code_blocks=[
                    CodeBlock(
                        content=(
                            "\ufeffdef check():\n"
                            "\treturn "
                            "\u201chealthy\u201d\n"
                        ),
                        language="python",
                    )
                ],
                links=[
                    DocumentLink(
                        text=(
                            "Azure\u00a0documentation"
                        ),
                        url=(
                            "https://learn.microsoft.com/"
                            "\u200bazure"
                        ),
                    )
                ],
            )
        ],
        metadata={
            "description": (
                "Azure\u00a0guide\u2026"
            ),
            "tags": [
                "cloud\u200b",
                "reliability\u2014azure",
            ],
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU UNICODE CLEANER")
    print("=" * 70)

    document = create_test_document()

    cleaner = CleanerRegistry.create(
        "unicode"
    )

    assert isinstance(
        cleaner,
        UnicodeCleaner,
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
        "\nHeading :",
        repr(section.heading),
    )

    print(
        "Paragraphes :",
        section.paragraphs,
    )

    print(
        "Code :",
        repr(
            section.code_blocks[0].content
        ),
    )

    print(
        "Métadonnées :",
        cleaned_document.metadata,
    )

    assert document.title == (
        "\ufeffAzure\u00a0Reliability\u200b"
    )

    assert cleaned_document.title == (
        "Azure Reliability"
    )

    assert cleaned_document.author == (
        "Microsoft Corporation"
    )

    assert cleaned_document.local_path == (
        "data/azure guide.md"
    )

    assert section.heading == (
        "Reliability-principles"
    )

    assert section.heading_path == [
        "Azure Architecture",
        "Reliability-principles",
    ]

    assert section.paragraphs[0] == (
        '"Azure" workloads must remain available...'
    )

    assert section.paragraphs[1] == (
        "Use availability zones "
        "and geo-replication."
    )

    assert section.paragraphs[2] == (
        "French accents remain: "
        "fiabilité, sécurité, résilience."
    )

    assert section.lists == [
        [
            "Use monitoring",
            "Protect data",
        ]
    ]

    assert (
        section.tables[0].headers
        == ["Service", "SLA value"]
    )

    assert (
        section.tables[0].rows
        == [
            [
                "Virtual Machines",
                "99.99 %",
            ]
        ]
    )

    assert (
        section.code_blocks[0].content
        == (
            "def check():\n"
            '\treturn "healthy"'
        )
    )

    assert (
        section.links[0].text
        == "Azure documentation"
    )

    assert (
        section.links[0].url
        == (
            "https://learn.microsoft.com/"
            "azure"
        )
    )

    assert (
        cleaned_document.metadata[
            "description"
        ]
        == "Azure guide..."
    )

    assert (
        cleaned_document.metadata[
            "tags"
        ]
        == [
            "cloud",
            "reliability-azure",
        ]
    )

    assert (
        "unicode_cleaner"
        in cleaned_document.metadata[
            "applied_cleaners"
        ]
    )

    print(
        "\n UnicodeCleaner fonctionnel."
    )


if __name__ == "__main__":
    main()