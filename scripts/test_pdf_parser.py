import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.source import SourceConfig
from src.parsing import (
    ParserRegistry,
    PdfParser,
)


def main() -> None:
    file_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parser_test"
        / "azure_security.pdf"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Le PDF de test n'existe pas. Exécute d'abord : "
            "python scripts\\create_test_pdf.py"
        )

    source = SourceConfig(
        id="pdf_test",
        name="PDF Test",
        type="local_folder",
        enabled=True,
        input={
            "path": str(file_path.parent),
        },
        metadata={
            "organization": "Microsoft",
            "domain": "Azure",
            "language": "en",
            "source_format": "pdf",
        },
    )

    print("=" * 70)
    print("TEST DU PDF PARSER")
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
        PdfParser,
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
        "Nombre total de pages :",
        document.metadata.get("page_count"),
    )

    print(
        "Pages avec texte :",
        document.metadata.get(
            "extracted_page_count"
        ),
    )

    print(
        "Statut extraction :",
        document.metadata.get(
            "extraction_status"
        ),
    )

    print(
        "OCR utilisé :",
        document.metadata.get("ocr_used"),
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
            f"Position    : {section.position}"
        )
        print(
            f"Paragraphes : {len(section.paragraphs)}"
        )

        for paragraph in section.paragraphs:
            print(
                f"  - {paragraph}"
            )

    assert document.title == (
        "Azure Security Guide"
    )

    assert document.document_format == "pdf"

    assert document.author == "Microsoft"

    assert document.language == "en"

    assert document.section_count == 2

    assert document.metadata["page_count"] == 2

    assert (
        document.metadata[
            "extracted_page_count"
        ]
        == 2
    )

    assert (
        document.metadata[
            "extraction_status"
        ]
        == "success"
    )

    assert (
        document.metadata["ocr_used"]
        is False
    )

    assert "Azure security protects" in (
        document.sections[0].text
    )

    assert "Incident response" in (
        document.sections[1].text
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parsed_azure_security.json"
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

    print("\n✅ PdfParser fonctionnel.")


if __name__ == "__main__":
    main()