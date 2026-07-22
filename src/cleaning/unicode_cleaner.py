from __future__ import annotations

import unicodedata

from src.cleaning.base import BaseCleaner
from src.cleaning.registry import CleanerRegistry
from src.models.document import ParsedDocument


@CleanerRegistry.register(
    "unicode",
    "unicode_cleaner",
)
class UnicodeCleaner(BaseCleaner):
    """
    Normalise les caractères Unicode problématiques.

    Ce cleaner :
    - remplace les espaces Unicode par des espaces standards ;
    - supprime les caractères invisibles ;
    - normalise les guillemets typographiques ;
    - normalise les tirets ;
    - conserve les lettres accentuées ;
    - applique une normalisation Unicode NFKC.
    """

    cleaner_name = "unicode_cleaner"

    CHARACTER_REPLACEMENTS = {
        # Espaces Unicode
        "\u00a0": " ",   # non-breaking space
        "\u1680": " ",
        "\u180e": "",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",

        # Caractères invisibles
        "\u200b": "",    # zero-width space
        "\u200c": "",    # zero-width non-joiner
        "\u200d": "",    # zero-width joiner
        "\u2060": "",    # word joiner
        "\ufeff": "",    # BOM
        "\u00ad": "",    # soft hyphen

        # Guillemets
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",

        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',

        # Tirets
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",

        # Ponctuation
        "\u2026": "...",
        "\u00b7": ".",
    }

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Applique la normalisation Unicode à toutes les
        zones textuelles du document.
        """

        document.title = self._clean_text(
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

        document.local_path = self._clean_text(
            document.local_path
        )

        for section in document.sections:
            section.heading = self._clean_optional_text(
                section.heading
            )

            section.heading_path = [
                self._clean_text(value)
                for value in section.heading_path
                if self._clean_text(value)
            ]

            section.paragraphs = [
                cleaned_paragraph
                for paragraph in section.paragraphs
                if (
                    cleaned_paragraph
                    := self._clean_text(paragraph)
                )
            ]

            cleaned_lists: list[list[str]] = []

            for items in section.lists:
                cleaned_items = [
                    cleaned_item
                    for item in items
                    if (
                        cleaned_item
                        := self._clean_text(item)
                    )
                ]

                if cleaned_items:
                    cleaned_lists.append(
                        cleaned_items
                    )

            section.lists = cleaned_lists

            for table in section.tables:
                table.headers = [
                    self._clean_text(header)
                    for header in table.headers
                ]

                cleaned_rows: list[list[str]] = []

                for row in table.rows:
                    cleaned_row = [
                        self._clean_text(cell)
                        for cell in row
                    ]

                    if any(cleaned_row):
                        cleaned_rows.append(
                            cleaned_row
                        )

                table.rows = cleaned_rows

            for code_block in section.code_blocks:
                code_block.content = (
                    self._clean_code_block(
                        code_block.content
                    )
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
                link.text = self._clean_text(
                    link.text
                )

                link.url = self._clean_text(
                    link.url
                )

            section.links = [
                link
                for link in section.links
                if link.url
            ]

        document.metadata = self._clean_metadata(
            document.metadata
        )

        return document

    @classmethod
    def _clean_text(
        cls,
        value: str | None,
    ) -> str:
        """
        Nettoie et normalise une chaîne de caractères.
        """

        if value is None:
            return ""

        normalized = str(value)

        normalized = unicodedata.normalize(
            "NFKC",
            normalized,
        )

        for source, replacement in (
            cls.CHARACTER_REPLACEMENTS.items()
        ):
            normalized = normalized.replace(
                source,
                replacement,
            )

        normalized = cls._remove_control_characters(
            normalized
        )

        return normalized.strip()

    @classmethod
    def _clean_code_block(
        cls,
        value: str | None,
    ) -> str:
        """
        Nettoie un bloc de code sans supprimer
        l'indentation interne.
        """

        if value is None:
            return ""

        normalized = str(value)

        normalized = unicodedata.normalize(
            "NFKC",
            normalized,
        )

        for source, replacement in (
            cls.CHARACTER_REPLACEMENTS.items()
        ):
            normalized = normalized.replace(
                source,
                replacement,
            )

        normalized = cls._remove_control_characters(
            normalized,
            preserve_newlines=True,
            preserve_tabs=True,
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

    @staticmethod
    def _remove_control_characters(
        value: str,
        preserve_newlines: bool = True,
        preserve_tabs: bool = False,
    ) -> str:
        """
        Supprime les caractères de contrôle inutiles.
        """

        cleaned_characters: list[str] = []

        for character in value:
            if (
                preserve_newlines
                and character in {"\n", "\r"}
            ):
                cleaned_characters.append(character)
                continue

            if (
                preserve_tabs
                and character == "\t"
            ):
                cleaned_characters.append(character)
                continue

            category = unicodedata.category(
                character
            )

            if category.startswith("C"):
                continue

            cleaned_characters.append(character)

        return "".join(cleaned_characters)

    @classmethod
    def _clean_optional_text(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Nettoie un champ facultatif.
        """

        cleaned_value = cls._clean_text(
            value
        )

        return cleaned_value or None

    @classmethod
    def _clean_metadata(
        cls,
        metadata: dict | None,
    ) -> dict:
        """
        Nettoie récursivement les chaînes présentes dans
        les métadonnées.
        """

        if not metadata:
            return {}

        return {
            cls._clean_text(str(key)): (
                cls._clean_metadata_value(value)
            )
            for key, value in metadata.items()
        }

    @classmethod
    def _clean_metadata_value(
        cls,
        value,
    ):
        """
        Nettoie récursivement une valeur de métadonnée.
        """

        if isinstance(value, str):
            return cls._clean_text(value)

        if isinstance(value, list):
            return [
                cls._clean_metadata_value(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls._clean_metadata_value(item)
                for item in value
            )

        if isinstance(value, dict):
            return cls._clean_metadata(value)

        return value