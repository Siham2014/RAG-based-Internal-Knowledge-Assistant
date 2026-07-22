import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.source import SourceConfig
from src.parsing import HtmlParser, ParserRegistry


def main() -> None:
    file_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parser_test"
        / "azure_reliability.html"
    )

    source = SourceConfig(
        id="html_test",
        name="HTML Test",
        type="local_folder",
        enabled=True,
        input={
            "path": str(file_path.parent),
        },
        metadata={
            "organization": "Microsoft",
            "domain": "Azure",
            "language": "en",
            "source_format": "html",
        },
    )

    print("=" * 70)
    print("TEST DU HTML PARSER")
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
        HtmlParser,
    )

    document = parser.parse(
        file_path
    )

    print(
        f"\nDocument ID : {document.document_id}"
    )
    print(
        f"Titre       : {document.title}"
    )
    print(
        f"Format      : {document.document_format}"
    )
    print(
        f"Sections    : {document.section_count}"
    )
    print(
        f"SHA-256     : {document.checksum_sha256}"
    )
    print(
        f"Auteur      : {document.author}"
    )
    print(
        f"Langue      : {document.language}"
    )
    print(
        f"URL source  : {document.source_url}"
    )

    print("\nStructure extraite :")

    for section in document.sections:
        print("\n" + "-" * 60)
        print(
            f"ID          : {section.section_id}"
        )
        print(
            f"Titre       : {section.heading}"
        )
        print(
            f"Niveau      : {section.heading_level}"
        )
        print(
            f"Chemin      : {section.heading_path}"
        )
        print(
            f"Paragraphes : {len(section.paragraphs)}"
        )
        print(
            f"Listes      : {len(section.lists)}"
        )
        print(
            f"Tableaux    : {len(section.tables)}"
        )
        print(
            f"Codes       : {len(section.code_blocks)}"
        )
        print(
            f"Liens       : {len(section.links)}"
        )

    assert document.title == "Azure Reliability"
    assert document.document_format == "html"
    assert document.section_count == 4

    assert document.author == "Microsoft"
    assert document.language == "en"

    assert document.source_url == (
        "https://learn.microsoft.com/"
        "azure/reliability/"
    )

    assert (
        document.sections[0].heading
        == "Azure Reliability"
    )

    assert len(
        document.sections[0].links
    ) == 1

    assert (
        document.sections[1].heading
        == "Recommendations"
    )

    assert (
        "Use availability zones"
        in document.sections[1].lists[0]
    )

    assert len(
        document.sections[2].tables
    ) == 1

    assert (
        document.sections[2]
        .tables[0]
        .headers
        == ["Service", "SLA"]
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

    assert "console.log" not in document.text

    output_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parsed_azure_reliability.json"
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

    print("\n HtmlParser fonctionnel.")


if __name__ == "__main__":
    main()