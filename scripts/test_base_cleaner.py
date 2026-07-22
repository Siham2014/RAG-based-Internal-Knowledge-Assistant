import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import (
    BaseCleaner,
    CleaningError,
)
from src.models.document import (
    DocumentSection,
    ParsedDocument,
)


class TestCleaner(BaseCleaner):
    """
    Cleaner temporaire utilisé uniquement pour le test.
    """

    cleaner_name = "test_cleaner"

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        document.title = (
            f"{document.title} - cleaned"
        )

        return document


def create_test_document() -> ParsedDocument:
    return ParsedDocument(
        document_id="cleaning_test:document",
        source_id="cleaning_test",
        source_type="local_folder",
        local_path="data/test.md",
        checksum_sha256="test-sha256",
        title="Azure Reliability",
        author="Microsoft",
        source_url=None,
        language="en",
        organization="Microsoft",
        document_format="markdown",
        sections=[
            DocumentSection(
                section_id=(
                    "cleaning_test:"
                    "document:section-0"
                ),
                heading="Azure Reliability",
                heading_level=1,
                position=0,
                heading_path=[
                    "Azure Reliability"
                ],
                paragraphs=[
                    "Azure workloads must be reliable."
                ],
            )
        ],
        metadata={
            "domain": "Azure",
        },
    )


def main() -> None:
    print("=" * 70)
    print("TEST DU BASE CLEANER")
    print("=" * 70)

    original_document = create_test_document()

    cleaner = TestCleaner()

    print(
        "\nDescription du cleaner :"
    )
    print(
        cleaner.describe()
    )

    cleaned_document = cleaner.run(
        original_document
    )

    print(
        "\nTitre original :",
        original_document.title,
    )

    print(
        "Titre nettoyé :",
        cleaned_document.title,
    )

    print(
        "Cleaners appliqués :",
        cleaned_document.metadata.get(
            "applied_cleaners"
        ),
    )

    assert original_document.title == (
        "Azure Reliability"
    )

    assert cleaned_document.title == (
        "Azure Reliability - cleaned"
    )

    assert (
        cleaned_document
        is not original_document
    )

    assert (
        cleaned_document.sections
        is not original_document.sections
    )

    assert (
        cleaned_document.metadata[
            "cleaning_applied"
        ]
        is True
    )

    assert (
        "test_cleaner"
        in cleaned_document.metadata[
            "applied_cleaners"
        ]
    )

    disabled_cleaner = TestCleaner(
        enabled=False
    )

    disabled_result = disabled_cleaner.run(
        original_document
    )

    assert disabled_result.title == (
        "Azure Reliability"
    )

    assert (
        disabled_result
        is not original_document
    )

    try:
        cleaner.run(
            "not a document"
        )

    except CleaningError as error:
        print(
            "\nErreur attendue :",
            error,
        )

    else:
        raise AssertionError(
            "Une CleaningError était attendue."
        )

    print(
        "\n✅ BaseCleaner fonctionnel."
    )


if __name__ == "__main__":
    main()