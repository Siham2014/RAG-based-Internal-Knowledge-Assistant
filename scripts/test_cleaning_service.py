import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import CleaningService
from src.models.document import (
    DocumentSection,
    ParsedDocument,
)


def create_document(
    document_id: str,
    title: str,
) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        source_id=document_id,
        source_type="html",
        local_path=(
            f"data/staging/{document_id}.html"
        ),
        checksum_sha256=(
            f"checksum-{document_id}"
        ),
        title=title,
        author="Microsoft",
        source_url=(
            "https://learn.microsoft.com/"
            f"{document_id}"
        ),
        language="en",
        organization="Microsoft",
        document_format="html",
        sections=[
            DocumentSection(
                section_id=(
                    f"{document_id}:section-0"
                ),
                heading=(
                    "Azure\u00a0Reliability"
                ),
                heading_level=1,
                position=0,
                heading_path=[
                    "Azure Reliability"
                ],
                paragraphs=[
                    (
                        "Azure   workloads "
                        "must remain available."
                    ),
                    (
                        "Azure workloads "
                        "must remain available."
                    ),
                    "Edit",
                    "Feedback",
                ],
                lists=[
                    [
                        "Previous",
                        "Use monitoring",
                        "Use monitoring",
                    ]
                ],
                tables=[],
                code_blocks=[],
                links=[],
            ),
            DocumentSection(
                section_id=(
                    f"{document_id}:section-1"
                ),
                heading="Table of contents",
                heading_level=2,
                position=1,
                heading_path=[
                    "Table of contents"
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
        ],
        metadata={
            "domain": "Azure",
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU CLEANING SERVICE")
    print("=" * 70)

    documents = [
        create_document(
            document_id="document-1",
            title="\ufeffAzure\u00a0Guide 1",
        ),
        create_document(
            document_id="document-2",
            title="\ufeffAzure\u00a0Guide 2",
        ),
        create_document(
            document_id="document-3",
            title="\ufeffAzure\u00a0Guide 3",
        ),
    ]

    service = CleaningService()

    print(
        "\nConfiguration :"
    )

    print(
        service.describe()
    )

    cleaned_documents, report = (
        service.clean_documents(
            documents
        )
    )

    print(
        "\nDocuments avant :",
        len(documents),
    )

    print(
        "Documents après :",
        len(cleaned_documents),
    )

    for document in cleaned_documents:
        print(
            "\nDocument :",
            document.document_id,
        )

        print(
            "Titre :",
            repr(document.title),
        )

        print(
            "Nombre de sections :",
            len(document.sections),
        )

        for section in document.sections:
            print(
                "Section :",
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
        "\nRapport global :"
    )

    print(
        report.to_dict()
    )

    # Les documents originaux restent inchangés.
    assert documents[0].title == (
        "\ufeffAzure\u00a0Guide 1"
    )

    assert len(
        documents[0].sections
    ) == 2

    # Les trois documents doivent réussir.
    assert len(
        cleaned_documents
    ) == 3

    assert report.success is True

    assert report.total_documents == 3

    assert (
        report.successful_documents
        == 3
    )

    assert report.failed_documents == 0

    assert report.input_sections == 6

    assert report.output_sections == 3

    assert report.removed_sections == 3

    for document in cleaned_documents:
        assert len(document.sections) == 1

        assert (
            "\ufeff"
            not in document.title
        )

        assert "\u00a0" not in (
            document.title
        )

        section = document.sections[0]

        assert section.heading == (
            "Azure Reliability"
        )

        assert section.paragraphs == [
            (
                "Azure workloads "
                "must remain available."
            )
        ]

        assert section.lists == [
            [
                "Use monitoring",
            ]
        ]

        assert (
            document.metadata[
                "cleaning_service_applied"
            ]
            is True
        )

        assert (
            document.metadata[
                "cleaning_service_name"
            ]
            == "default_cleaning_service"
        )

    assert len(report.results) == 3

    assert all(
        result.success
        for result in report.results
    )

    print(
        "\nCleaningService fonctionnel."
    )


if __name__ == "__main__":
    main()