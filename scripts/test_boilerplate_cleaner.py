import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import (
    BoilerplateCleaner,
    CleanerRegistry,
)
from src.models.document import (
    DocumentLink,
    DocumentSection,
    ParsedDocument,
)


def create_test_document() -> ParsedDocument:
    return ParsedDocument(
        document_id="boilerplate_test:document",
        source_id="boilerplate_test",
        source_type="html",
        local_path=(
            "data/staging/"
            "azure_reliability.html"
        ),
        checksum_sha256="boilerplate-test",
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
                    "boilerplate_test:"
                    "document:section-0"
                ),
                heading="Azure Reliability",
                heading_level=1,
                position=0,
                heading_path=[
                    "Azure Reliability"
                ],
                paragraphs=[
                    (
                        "Azure reliability helps "
                        "workloads remain available."
                    ),
                    "Edit",
                    "Feedback",
                    "Last updated: 2026-07-01",
                    "Was this page helpful?",
                ],
                lists=[
                    [
                        "Use availability zones",
                        "Previous",
                        "Next",
                    ]
                ],
                tables=[],
                code_blocks=[],
                links=[
                    DocumentLink(
                        text="Azure Architecture Center",
                        url=(
                            "https://learn.microsoft.com/"
                            "azure/architecture/"
                        ),
                    ),
                    DocumentLink(
                        text="Edit on GitHub",
                        url=(
                            "https://github.com/"
                            "MicrosoftDocs/azure-docs"
                        ),
                    ),
                    DocumentLink(
                        text="Feedback",
                        url=(
                            "https://learn.microsoft.com/"
                            "feedback"
                        ),
                    ),
                ],
            ),
            DocumentSection(
                section_id=(
                    "boilerplate_test:"
                    "document:section-1"
                ),
                heading="Table of contents",
                heading_level=2,
                position=1,
                heading_path=[
                    "Azure Reliability",
                    "Table of contents",
                ],
                paragraphs=[
                    "Previous",
                    "Next",
                ],
                lists=[],
                tables=[],
                code_blocks=[],
                links=[],
            ),
            DocumentSection(
                section_id=(
                    "boilerplate_test:"
                    "document:section-2"
                ),
                heading="Resiliency",
                heading_level=2,
                position=2,
                heading_path=[
                    "Azure Reliability",
                    "Resiliency",
                ],
                paragraphs=[
                    (
                        "Resiliency is the ability "
                        "to recover from failures."
                    )
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
    print("TEST DU BOILERPLATE CLEANER")
    print("=" * 70)

    document = create_test_document()

    cleaner = CleanerRegistry.create(
        "boilerplate"
    )

    assert isinstance(
        cleaner,
        BoilerplateCleaner,
    )

    cleaned_document = cleaner.run(
        document
    )

    print(
        "\nNombre de sections avant :",
        len(document.sections),
    )

    print(
        "Nombre de sections après :",
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
        "boilerplate_removed"
    ]

    print(
        "\nRapport :",
        report,
    )

    assert len(document.sections) == 3

    assert len(
        cleaned_document.sections
    ) == 2

    first_section = (
        cleaned_document.sections[0]
    )

    assert first_section.heading == (
        "Azure Reliability"
    )

    assert first_section.paragraphs == [
        (
            "Azure reliability helps "
            "workloads remain available."
        )
    ]

    assert first_section.lists == [
        [
            "Use availability zones"
        ]
    ]

    assert len(
        first_section.links
    ) == 1

    assert (
        first_section.links[0].text
        == "Azure Architecture Center"
    )

    assert (
        cleaned_document.sections[1].heading
        == "Resiliency"
    )

    assert report["paragraphs"] == 4
    assert report["list_items"] == 2
    assert report["links"] == 2
    assert report["sections"] == 1

    assert (
        "boilerplate_cleaner"
        in cleaned_document.metadata[
            "applied_cleaners"
        ]
    )

    print(
        "\n BoilerplateCleaner fonctionnel."
    )


if __name__ == "__main__":
    main()