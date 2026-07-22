from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.models.document import (
    DocumentSection,
    ParsedDocument,
)
from src.parsing.base import BaseParser, ParserError
from src.parsing.registry import ParserRegistry


@ParserRegistry.register(".pdf")
class PdfParser(BaseParser):
    """
    Parser pour les PDF contenant une couche texte.

    Cette première version n'utilise pas d'OCR.

    Chaque page contenant du texte devient une section du
    ParsedDocument afin de conserver le numéro de page.
    """

    def parse(
        self,
        file_path: Path,
    ) -> ParsedDocument:
        """
        Parse un fichier PDF et retourne un ParsedDocument.
        """

        file_path = file_path.resolve()

        self.validate_file(file_path)

        reader = self._open_pdf(file_path)

        document_id = self.build_document_id(
            file_path
        )

        pdf_metadata = self._extract_pdf_metadata(
            reader
        )

        sections = self._extract_sections(
            reader=reader,
            document_id=document_id,
        )

        extracted_page_count = len(sections)
        total_page_count = len(reader.pages)

        title = self._extract_title(
            pdf_metadata=pdf_metadata,
            sections=sections,
            file_path=file_path,
        )

        source_metadata = self._get_source_metadata()

        extraction_status = (
            "success"
            if extracted_page_count > 0
            else "no_text"
        )

        return ParsedDocument(
            document_id=document_id,
            source_id=self.source.id,
            source_type=self.source.type,
            local_path=str(file_path),
            checksum_sha256=self.compute_sha256(
                file_path
            ),
            title=title,
            author=(
                pdf_metadata.get("author")
                or source_metadata.get("author")
            ),
            source_url=self._get_source_url(),
            language=source_metadata.get("language"),
            organization=source_metadata.get(
                "organization"
            ),
            document_format="pdf",
            sections=sections,
            metadata={
                **source_metadata,
                **pdf_metadata,
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "parser": self.__class__.__name__,
                "page_count": total_page_count,
                "extracted_page_count": extracted_page_count,
                "section_count": len(sections),
                "extraction_status": extraction_status,
                "ocr_used": False,
            },
        )

    @staticmethod
    def _open_pdf(
        file_path: Path,
    ) -> PdfReader:
        """
        Ouvre le PDF avec pypdf.
        """

        try:
            reader = PdfReader(
                str(file_path)
            )

        except PdfReadError as error:
            raise ParserError(
                f"Le fichier PDF est invalide ou endommagé : "
                f"{file_path}"
            ) from error

        except OSError as error:
            raise ParserError(
                f"Impossible d'ouvrir le fichier PDF "
                f"{file_path} : {error}"
            ) from error

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")

            except Exception as error:
                raise ParserError(
                    "Le PDF est chiffré et ne peut pas être "
                    f"ouvert : {file_path}"
                ) from error

            if result == 0:
                raise ParserError(
                    "Le PDF est protégé par un mot de passe : "
                    f"{file_path}"
                )

        return reader

    def _extract_sections(
        self,
        reader: PdfReader,
        document_id: str,
    ) -> list[DocumentSection]:
        """
        Extrait le texte page par page.

        Une page vide n'est pas ajoutée aux sections.
        """

        sections: list[DocumentSection] = []

        for page_index, page in enumerate(
            reader.pages
        ):
            page_number = page_index + 1

            try:
                raw_text = page.extract_text() or ""

            except Exception as error:
                raise ParserError(
                    "Erreur pendant l'extraction de la page "
                    f"{page_number}."
                ) from error

            cleaned_text = self._clean_page_text(
                raw_text
            )

            if not cleaned_text:
                continue

            paragraphs = self._split_paragraphs(
                cleaned_text
            )

            if not paragraphs:
                continue

            section_position = len(sections)

            section = DocumentSection(
                section_id=(
                    f"{document_id}:page-{page_number}"
                ),
                heading=f"Page {page_number}",
                heading_level=None,
                position=section_position,
                heading_path=[
                    f"Page {page_number}"
                ],
                paragraphs=paragraphs,
            )

            sections.append(section)

        return sections

    @staticmethod
    def _clean_page_text(
        text: str,
    ) -> str:
        """
        Nettoie le texte extrait d'une page.

        Les retours à la ligne multiples sont conservés pour
        permettre la détection des paragraphes.
        """

        if not text:
            return ""

        normalized = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        normalized = normalized.replace(
            "\x00",
            "",
        )

        normalized = re.sub(
            r"[ \t]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r" *\n *",
            "\n",
            normalized,
        )

        normalized = re.sub(
            r"\n{3,}",
            "\n\n",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _split_paragraphs(
        text: str,
    ) -> list[str]:
        """
        Divise le texte en paragraphes.

        Si pypdf ne fournit pas de lignes vides, les lignes
        sont regroupées dans un paragraphe unique.
        """

        blocks = re.split(
            r"\n\s*\n",
            text,
        )

        paragraphs: list[str] = []

        for block in blocks:
            cleaned_block = block.strip()

            if not cleaned_block:
                continue

            lines = [
                line.strip()
                for line in cleaned_block.splitlines()
                if line.strip()
            ]

            paragraph = " ".join(lines)

            paragraph = re.sub(
                r"\s+",
                " ",
                paragraph,
            ).strip()

            if paragraph:
                paragraphs.append(paragraph)

        return paragraphs

    @staticmethod
    def _extract_pdf_metadata(
        reader: PdfReader,
    ) -> dict[str, Any]:
        """
        Extrait les métadonnées natives du PDF.
        """

        metadata: dict[str, Any] = {}

        raw_metadata = reader.metadata

        if raw_metadata is None:
            return metadata

        metadata_mapping = {
            "/Title": "pdf_title",
            "/Author": "author",
            "/Subject": "subject",
            "/Creator": "creator",
            "/Producer": "producer",
            "/CreationDate": "creation_date",
            "/ModDate": "modification_date",
            "/Keywords": "keywords",
        }

        for pdf_key, output_key in metadata_mapping.items():
            value = raw_metadata.get(
                pdf_key
            )

            if value is None:
                continue

            cleaned_value = str(value).strip()

            if cleaned_value:
                metadata[output_key] = cleaned_value

        return metadata

    @staticmethod
    def _extract_title(
        pdf_metadata: dict[str, Any],
        sections: list[DocumentSection],
        file_path: Path,
    ) -> str:
        """
        Priorité pour déterminer le titre :

        1. titre des métadonnées PDF ;
        2. première ligne textuelle du document ;
        3. nom du fichier.
        """

        metadata_title = pdf_metadata.get(
            "pdf_title"
        )

        if metadata_title:
            return str(metadata_title).strip()

        if sections:
            first_section = sections[0]

            if first_section.paragraphs:
                first_paragraph = (
                    first_section.paragraphs[0]
                )

                first_line = first_paragraph.split(
                    ".",
                    maxsplit=1,
                )[0].strip()

                if 3 <= len(first_line) <= 150:
                    return first_line

        return file_path.stem.replace(
            "-",
            " ",
        ).replace(
            "_",
            " ",
        ).strip().title()

    def _get_source_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Récupère les métadonnées définies dans sources.yaml.
        """

        metadata = getattr(
            self.source,
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            return dict(metadata)

        return {}

    def _get_source_url(
        self,
    ) -> str | None:
        """
        Récupère l'URL éventuelle de la source.
        """

        input_config = getattr(
            self.source,
            "input",
            {},
        )

        if not isinstance(input_config, dict):
            return None

        url = (
            input_config.get("url")
            or input_config.get("source_url")
        )

        return str(url) if url else None