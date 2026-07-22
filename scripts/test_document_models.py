import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.document import (
    CodeBlock,
    DocumentLink,
    DocumentSection,
    DocumentTable,
    ParsedDocument,
)


def main() -> None:
    introduction = DocumentSection(
        section_id="azure-retry-pattern:section-0",
        heading="Retry Pattern",
        heading_level=1,
        paragraphs=[
            (
                "The Retry pattern enables an application "
                "to handle transient failures."
            )
        ],
        links=[
            DocumentLink(
                text="Azure Architecture Center",
                url=(
                    "https://learn.microsoft.com/"
                    "azure/architecture/"
                ),
            )
        ],
        position=0,
        heading_path=[
            "Cloud Design Patterns",
            "Retry Pattern",
        ],
    )

    implementation = DocumentSection(
        section_id="azure-retry-pattern:section-1",
        heading="Implementation",
        heading_level=2,
        paragraphs=[
            "Retry failed operations with controlled delays."
        ],
        lists=[
            [
                "Use exponential backoff",
                "Limit the number of retries",
                "Log each final failure",
            ]
        ],
        tables=[
            DocumentTable(
                headers=[
                    "Attempt",
                    "Delay",
                ],
                rows=[
                    ["1", "1 second"],
                    ["2", "2 seconds"],
                    ["3", "4 seconds"],
                ],
                caption="Example retry delays",
            )
        ],
        code_blocks=[
            CodeBlock(
                language="python",
                content=(
                    "for attempt in range(3):\n"
                    "    execute_operation()"
                ),
            )
        ],
        position=1,
        heading_path=[
            "Cloud Design Patterns",
            "Retry Pattern",
            "Implementation",
        ],
    )

    document = ParsedDocument(
        document_id="azure-retry-pattern",
        source_id="azure_architecture_patterns",
        source_type="html",
        local_path=(
            "data/raw/html/"
            "azure-retry-pattern.html"
        ),
        checksum_sha256="abc123",
        title="Retry Pattern",
        author="Microsoft",
        source_url=(
            "https://learn.microsoft.com/en-us/"
            "azure/architecture/patterns/retry"
        ),
        language="en",
        organization="Microsoft",
        document_format="html",
        metadata={
            "domain": "Azure",
        },
    )

    document.add_section(introduction)
    document.add_section(implementation)

    print("=" * 70)
    print("TEST DES MODÈLES DE DOCUMENT")
    print("=" * 70)

    print(f"Document : {document.title}")
    print(f"Sections : {document.section_count}")
    print(f"Vide     : {document.is_empty}")

    print("\nTexte reconstruit :\n")
    print(document.text)

    serialized = document.to_dict()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "test_parsed_document.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            serialized,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    restored_document = ParsedDocument.from_dict(
        json.loads(
            output_path.read_text(
                encoding="utf-8"
            )
        )
    )

    assert restored_document.document_id == (
        document.document_id
    )

    assert restored_document.section_count == 2

    assert (
        restored_document.sections[1]
        .code_blocks[0]
        .language
        == "python"
    )

    assert "exponential backoff" in (
        restored_document.text
    )

    print("\nFichier JSON créé :")
    print(output_path)

    print("\n Tous les tests sont réussis.")


if __name__ == "__main__":
    main()