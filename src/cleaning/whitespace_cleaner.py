from __future__ import annotations

import re

from src.cleaning.base import BaseCleaner
from src.cleaning.registry import CleanerRegistry
from src.models.document import ParsedDocument


@CleanerRegistry.register(
    "whitespace",
    "whitespace_cleaner",
)
class WhitespaceCleaner(BaseCleaner):
    """
    Nettoie les espaces inutiles dans un ParsedDocument.

    Ce cleaner :
    - supprime les espaces au début et à la fin ;
    - remplace les espaces multiples par un seul espace ;
    - normalise les retours à la ligne ;
    - retire les paragraphes, listes et liens vides ;
    - conserve l'indentation interne des blocs de code.
    """

    cleaner_name = "whitespace_cleaner"

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Nettoie toutes les zones textuelles du document.
        """

        document.title = self._clean_inline_text(
            document.title
        )

        document.author = self._clean_optional_text(
            document.author
        )

        document.source_url = self._clean_optional_text(
            document.source_url
        )

        document.language = self._clean_optional_text(
            document.language
        )

        document.organization = self._clean_optional_text(
            document.organization
        )

        for section in document.sections:
            section.heading = self._clean_optional_text(
                section.heading
            )

            section.heading_path = [
                cleaned_value
                for value in section.heading_path
                if (
                    cleaned_value
                    := self._clean_inline_text(value)
                )
            ]

            section.paragraphs = [
                cleaned_paragraph
                for paragraph in section.paragraphs
                if (
                    cleaned_paragraph
                    := self._clean_multiline_text(
                        paragraph
                    )
                )
            ]

            cleaned_lists: list[list[str]] = []

            for items in section.lists:
                cleaned_items = [
                    cleaned_item
                    for item in items
                    if (
                        cleaned_item
                        := self._clean_inline_text(item)
                    )
                ]

                if cleaned_items:
                    cleaned_lists.append(
                        cleaned_items
                    )

            section.lists = cleaned_lists

            for table in section.tables:
                table.headers = [
                    self._clean_inline_text(header)
                    for header in table.headers
                ]

                cleaned_rows: list[list[str]] = []

                for row in table.rows:
                    cleaned_row = [
                        self._clean_inline_text(cell)
                        for cell in row
                    ]

                    if any(cleaned_row):
                        cleaned_rows.append(
                            cleaned_row
                        )

                table.rows = cleaned_rows

            for code_block in section.code_blocks:
                code_block.content = self._clean_code_block(
                    code_block.content
                )

                code_block.language = (
                    self._clean_optional_text(
                        code_block.language
                    )
                )

            section.code_blocks = [
                code_block
                for code_block in section.code_blocks
                if code_block.content
            ]

            for link in section.links:
                link.text = self._clean_inline_text(
                    link.text
                )

                link.url = self._clean_inline_text(
                    link.url
                )

            section.links = [
                link
                for link in section.links
                if link.url
            ]

        return document

    @staticmethod
    def _clean_inline_text(
        value: str | None,
    ) -> str:
        """
        Nettoie une chaîne sur une seule ligne.

        Exemple :

        "  Azure    Reliability  "

        devient :

        "Azure Reliability"
        """

        if value is None:
            return ""

        normalized = str(value)

        normalized = normalized.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        normalized = normalized.replace(
            "\n",
            " ",
        )

        normalized = re.sub(
            r"[ \t\f\v]+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _clean_multiline_text(
        value: str | None,
    ) -> str:
        """
        Nettoie un paragraphe pouvant contenir plusieurs
        lignes.

        Les retours doubles sont conservés lorsqu'ils
        représentent une séparation logique.
        """

        if value is None:
            return ""

        normalized = str(value)

        normalized = normalized.replace(
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
            r"[ \t\f\v]+",
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

        lines = [
            line.strip()
            for line in normalized.splitlines()
        ]

        cleaned_lines: list[str] = []
        previous_line_empty = False

        for line in lines:
            if not line:
                if (
                    cleaned_lines
                    and not previous_line_empty
                ):
                    cleaned_lines.append("")

                previous_line_empty = True
                continue

            cleaned_lines.append(line)
            previous_line_empty = False

        return "\n".join(
            cleaned_lines
        ).strip()

    @staticmethod
    def _clean_code_block(
        value: str | None,
    ) -> str:
        """
        Nettoie un bloc de code sans supprimer son
        indentation interne.

        Seuls les retours Windows, les caractères nuls,
        les espaces de fin de ligne et les lignes vides
        extérieures sont retirés.
        """

        if value is None:
            return ""

        normalized = str(value)

        normalized = normalized.replace(
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

        lines = [
            line.rstrip()
            for line in normalized.splitlines()
        ]

        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)

    @classmethod
    def _clean_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Nettoie un champ facultatif.

        Retourne None lorsque le contenu final est vide.
        """

        cleaned_value = cls._clean_inline_text(
            value
        )

        return cleaned_value or None