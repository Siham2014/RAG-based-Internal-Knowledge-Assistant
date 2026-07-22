import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import CleaningPipeline
from src.models.document import (
    DocumentLink,
    DocumentSection,
    ParsedDocument,
)


def create_test_document() -> ParsedDocument:
    return ParsedDocument(
        document_id="pipeline_test:document",
        source_id="pipeline_test",
        source_type="html",
        local_path=(
            "data/\ufeffazure\u00a0guide.html"
        ),
        checksum_sha256="pipeline-test",
        title=(
            "\ufeffAzure\u00a0"
            "Reliability\u200b"
        ),
        author=(
            "Microsoft\u202fCorporation"
        ),
        source_url=(
            "https://learn.microsoft.com/"
            "azure/reliability"
        ),
        language="en",
        organization="Microsoft",
        document_format="html",
        sections=[
            DocumentSection(
                section_id=(
                    "pipeline_test:"
                    "document:section-0"
                ),
                heading=(
                    "Reliability\u2014principles"
                ),
                heading_level=1,
                position=0,
                heading_path=[
                    "Azure\u00a0Architecture",
                    "Reliability\u2014principles",
                ],
                paragraphs=[
                    (
                        "\u201cAzure\u201d workloads "
                        "must remain available\u2026"
                    ),
                    (
                        "Azure   workloads must "
                        "remain available..."
                    ),
                    "Edit",
                    "Feedback",
                    (
                        "Use availability\u00a0zones."
                    ),
                    (
                        "Use availability zones."
                    ),
                ],
                lists=[
                    [
                        "Configure monitoring",
                        "Configure   monitoring",
                        "Previous",
                        "Next",
                    ]
                ],
                tables=[],
                code_blocks=[],
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
                    "pipeline_test:"
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
                    "pipeline_test:"
                    "document:section-2"
                ),
                heading="Monitoring",
                heading_level=2,
                position=2,
                heading_path=[
                    "Azure Reliability",
                    "Monitoring",
                ],
                paragraphs=[
                    (
                        "Configure monitoring "
                        "for critical services."
                    )
                ],
                lists=[],
                tables=[],
                code_blocks=[],
                links=[],
            ),
        ],
        metadata={
            "description": (
                "Azure\u00a0guide\u2026"
            ),
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU CLEANING PIPELINE")
    print("=" * 70)

    original_document = (
        create_test_document()
    )

    pipeline = CleaningPipeline()

    print(
        "\nConfiguration du pipeline :"
    )

    for cleaner in pipeline.describe()[
        "cleaners"
    ]:
        print(
            "-",
            cleaner["cleaner_name"],
            "| enabled =",
            cleaner["enabled"],
            "| config =",
            cleaner["config"],
        )

    cleaned_document, report = (
        pipeline.run(original_document)
    )

    print(
        "\nTitre original :",
        repr(original_document.title),
    )

    print(
        "Titre nettoyé :",
        repr(cleaned_document.title),
    )

    print(
        "\nSections avant :",
        len(original_document.sections),
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
        "\nRapport du pipeline :"
    )

    print(
        report.to_dict()
    )

    # Le document original ne doit pas être modifié.
    assert original_document.title == (
        "\ufeffAzure\u00a0"
        "Reliability\u200b"
    )

    assert len(
        original_document.sections
    ) == 3

    # Vérification Unicode.
    assert cleaned_document.title == (
        "Azure Reliability"
    )

    assert cleaned_document.author == (
        "Microsoft Corporation"
    )

    # La section Table of contents doit disparaître.
    assert len(
        cleaned_document.sections
    ) == 2

    first_section = (
        cleaned_document.sections[0]
    )

    assert first_section.heading == (
        "Reliability-principles"
    )

    assert first_section.paragraphs == [
        (
            '"Azure" workloads '
            "must remain available..."
        ),
        (
            "Azure workloads must "
            "remain available..."
        ),
        "Use availability zones.",
    ]

    assert first_section.lists == [
        [
            "Configure monitoring",
        ]
    ]

    assert len(
        first_section.links
    ) == 1

    assert (
        first_section.links[0].text
        == "Azure documentation"
    )

    second_section = (
        cleaned_document.sections[1]
    )

    assert second_section.heading == (
        "Monitoring"
    )

    assert second_section.paragraphs == [
        (
            "Configure monitoring "
            "for critical services."
        )
    ]

    # Vérification du rapport.
    assert report.success is True

    assert report.pipeline_name == (
        "default_cleaning_pipeline"
    )

    assert report.input_section_count == 3
    assert report.output_section_count == 2
    assert report.removed_section_count == 1

    assert len(report.cleaners) == 4

    assert [
        cleaner.cleaner_name
        for cleaner in report.cleaners
    ] == [
        "unicode_cleaner",
        "whitespace_cleaner",
        "boilerplate_cleaner",
        "duplicate_cleaner",
    ]

    assert all(
        cleaner.success
        for cleaner in report.cleaners
    )

    # Vérification des métadonnées.
    assert (
        cleaned_document.metadata[
            "cleaning_pipeline_applied"
        ]
        is True
    )

    assert (
        cleaned_document.metadata[
            "cleaning_pipeline_name"
        ]
        == "default_cleaning_pipeline"
    )

    assert (
        cleaned_document.metadata[
            "applied_cleaners"
        ]
        == [
            "unicode_cleaner",
            "whitespace_cleaner",
            "boilerplate_cleaner",
            "duplicate_cleaner",
        ]
    )

    print(
        "\nCleaningPipeline fonctionnel."
    )


if __name__ == "__main__":
    main()