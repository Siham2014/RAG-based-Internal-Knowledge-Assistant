import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.source import SourceConfig
from src.parsing import ParserRegistry
from src.parsing.markdown_parser import MarkdownParser


def main() -> None:
    file_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parser_test"
        / "azure_retry.md"
    )

    source = SourceConfig(
        id="markdown_test",
        name="Markdown Test",
        type="local_folder",
        enabled=True,
        input={
            "path": str(file_path.parent),
        },
        metadata={
            "organization": "Microsoft",
            "domain": "Azure",
            "language": "en",
            "source_format": "markdown",
        },
    )

    print("=" * 70)
    print("TEST DU MARKDOWN PARSER")
    print("=" * 70)

    print(
        "Extensions enregistrées :",
        ParserRegistry.available_extensions(),
    )

    parser = ParserRegistry.create(
        file_path=file_path,
        source=source,
    )

    assert isinstance(
        parser,
        MarkdownParser,
    )

    document = parser.parse(file_path)

    print(f"\nDocument ID : {document.document_id}")
    print(f"Titre       : {document.title}")
    print(f"Format      : {document.document_format}")
    print(f"Sections    : {document.section_count}")
    print(f"SHA-256     : {document.checksum_sha256}")

    print("\nStructure extraite :")

    for section in document.sections:
        print("\n" + "-" * 60)
        print(f"ID       : {section.section_id}")
        print(f"Titre    : {section.heading}")
        print(f"Niveau   : {section.heading_level}")
        print(f"Chemin   : {section.heading_path}")
        print(f"Paragraphes : {len(section.paragraphs)}")
        print(f"Listes      : {len(section.lists)}")
        print(f"Tableaux    : {len(section.tables)}")
        print(f"Codes       : {len(section.code_blocks)}")
        print(f"Liens       : {len(section.links)}")

    assert document.title == "Azure Retry Pattern"
    assert document.section_count == 4

    assert (
        document.sections[1].heading
        == "Recommendations"
    )

    assert (
        "Use exponential backoff"
        in document.sections[1].lists[0]
    )

    assert len(
        document.sections[2].tables
    ) == 1

    assert (
        document.sections[2]
        .tables[0]
        .headers
        == ["Attempt", "Delay"]
    )

    assert len(
        document.sections[3].code_blocks
    ) == 1

    assert (
        document.sections[3]
        .code_blocks[0]
        .language
        == "python"
    )

    assert any(
        link.url.startswith(
            "https://learn.microsoft.com"
        )
        for link in document.sections[0].links
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parsed_azure_retry.json"
    )

    output_path.write_text(
        json.dumps(
            document.to_dict(),
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nTexte reconstruit :\n")
    print(document.text)

    print("\nJSON produit :")
    print(output_path)

    print("\n MarkdownParser fonctionnel.")


if __name__ == "__main__":
    main()