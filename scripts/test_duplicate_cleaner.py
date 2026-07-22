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
    DuplicateCleaner,
)
from src.models.document import (
    CodeBlock,
    DocumentLink,
    DocumentSection,
    DocumentTable,
    ParsedDocument,
)


def create_test_document() -> ParsedDocument:
    repeated_paragraph = (
        "Azure availability zones improve "
        "application resilience."
    )

    return ParsedDocument(
        document_id="duplicate_test:document",
        source_id="duplicate_test",
        source_type="html",
        local_path=(
            "data/staging/"
            "duplicate_test.html"
        ),
        checksum_sha256="duplicate-test",
        title="Azure Reliability",
        author="Microsoft",
        source_url=(
            "https://learn.microsoft.com/"
            "azure/reliability/"
        ),
        language="en",
        organization="Microsoft",
        document_format="html",
        sections=[
            DocumentSection(
                section_id=(
                    "duplicate_test:"
                    "document:section-0"
                ),
                heading="Availability",
                heading_level=1,
                position=0,
                heading_path=[
                    "Availability"
                ],
                paragraphs=[
                    repeated_paragraph,
                    repeated_paragraph,
                    (
                        "Azure   availability zones "
                        "improve application resilience."
                    ),
                    "Monitor application health.",
                ],
                lists=[
                    [
                        "Use availability zones",
                        "Use availability zones",
                        "Configure monitoring",
                    ]
                ],
                tables=[
                    DocumentTable(
                        headers=[
                            "Service",
                            "SLA",
                        ],
                        rows=[
                            [
                                "Virtual Machines",
                                "99.99%",
                            ],
                            [
                                "Virtual Machines",
                                "99.99%",
                            ],
                            [
                                "App Service",
                                "99.95%",
                            ],
                        ],
                    )
                ],
                code_blocks=[
                    CodeBlock(
                        content=(
                            "def health_check():\n"
                            "    return True"
                        ),
                        language="python",
                    ),
                    CodeBlock(
                        content=(
                            "def health_check():\n"
                            "    return True"
                        ),
                        language="python",
                    ),
                ],
                links=[
                    DocumentLink(
                        text="Azure documentation",
                        url=(
                            "https://learn.microsoft.com/"
                            "azure"
                        ),
                    ),
                    DocumentLink(
                        text="Azure documentation",
                        url=(
                            "https://learn.microsoft.com/"
                            "azure"
                        ),
                    ),
                ],
            ),
            DocumentSection(
                section_id=(
                    "duplicate_test:"
                    "document:section-1"
                ),
                heading="Monitoring",
                heading_level=1,
                position=1,
                heading_path=[
                    "Monitoring"
                ],
                paragraphs=[
                    "Monitor application health.",
                    "Configure alerts for failures.",
                ],
                lists=[],
                tables=[],
                code_blocks=[],
                links=[],
            ),
            DocumentSection(
                section_id=(
                    "duplicate_test:"
                    "document:section-2"
                ),
                heading="Monitoring",
                heading_level=1,
                position=2,
                heading_path=[
                    "Monitoring"
                ],
                paragraphs=[
                    "Configure alerts for failures.",
                ],
                lists=[],
                tables=[],
                code_blocks=[],
                links=[],
            ),
        ],
        metadata={
            "domain": "Azure",
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU DUPLICATE CLEANER")
    print("=" * 70)

    document = create_test_document()

    cleaner = CleanerRegistry.create(
        "duplicate"
    )

    assert isinstance(
        cleaner,
        DuplicateCleaner,
    )

    cleaned_document = cleaner.run(
        document
    )

    print(
        "\nSections avant :",
        len(document.sections),
    )

    print(
        "Sections après :",
        len(cleaned_document.sections),
    )

    for section in cleaned_document.sections:
        print(
            "\nSection :",
            section.heading,
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
            "Tables :",
            [
                table.rows
                for table in section.tables
            ],
        )

        print(
            "Codes :",
            [
                code.content
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

    report = cleaned_document.metadata[
        "duplicates_removed"
    ]

    print(
        "\nRapport des doublons :",
        report,
    )

    # L'objet original ne doit pas être modifié.
    assert len(document.sections) == 3

    # La troisième section devient vide et doit être supprimée.
    assert len(cleaned_document.sections) == 2

    first_section = (
        cleaned_document.sections[0]
    )

    assert first_section.heading == (
        "Availability"
    )

    assert first_section.paragraphs == [
        (
            "Azure availability zones improve "
            "application resilience."
        ),
        "Monitor application health.",
    ]

    assert first_section.lists == [
        [
            "Use availability zones",
            "Configure monitoring",
        ]
    ]

    assert (
        first_section.tables[0].rows
        == [
            [
                "Virtual Machines",
                "99.99%",
            ],
            [
                "App Service",
                "99.95%",
            ],
        ]
    )

    assert len(
        first_section.code_blocks
    ) == 1

    assert len(
        first_section.links
    ) == 1

    second_section = (
        cleaned_document.sections[1]
    )

    assert second_section.heading == (
        "Monitoring"
    )

    assert second_section.paragraphs == [
        "Configure alerts for failures."
    ]

    assert report == {
        "paragraphs": 4,
        "list_items": 1,
        "table_rows": 1,
        "code_blocks": 1,
        "links": 1,
        "sections": 1,
    }

    assert (
        "duplicate_cleaner"
        in cleaned_document.metadata[
            "applied_cleaners"
        ]
    )

    config = cleaned_document.metadata[
        "duplicate_cleaner_config"
    ]

    assert config["case_sensitive"] is False
    assert config["global_scope"] is True

    print(
        "\nDuplicateCleaner fonctionnel."
    )


if __name__ == "__main__":
    main()