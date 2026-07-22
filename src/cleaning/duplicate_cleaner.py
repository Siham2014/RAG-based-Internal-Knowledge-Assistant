from __future__ import annotations

import hashlib
import re
from typing import Any

from src.cleaning.base import BaseCleaner
from src.cleaning.registry import CleanerRegistry
from src.models.document import ParsedDocument


@CleanerRegistry.register(
    "duplicate",
    "duplicates",
    "duplicate_cleaner",
)
class DuplicateCleaner(BaseCleaner):
    """
    Supprime les contenus dupliqués dans un ParsedDocument.

    Le cleaner peut supprimer les doublons présents dans :

    - les paragraphes ;
    - les éléments de listes ;
    - les lignes de tableaux ;
    - les blocs de code ;
    - les liens ;
    - les sections ;
    - les sections devenues vides après nettoyage.

    La première occurrence est toujours conservée.
    """

    cleaner_name = "duplicate_cleaner"

    def __init__(
        self,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            enabled=enabled,
            config=config,
        )

        self.case_sensitive = self.get_config(
            "case_sensitive",
            False,
        )

        self.global_scope = self.get_config(
            "global_scope",
            True,
        )

        self.remove_duplicate_paragraphs = self.get_config(
            "remove_duplicate_paragraphs",
            True,
        )

        self.remove_duplicate_list_items = self.get_config(
            "remove_duplicate_list_items",
            True,
        )

        self.remove_duplicate_table_rows = self.get_config(
            "remove_duplicate_table_rows",
            True,
        )

        self.remove_duplicate_code_blocks = self.get_config(
            "remove_duplicate_code_blocks",
            True,
        )

        self.remove_duplicate_links = self.get_config(
            "remove_duplicate_links",
            True,
        )

        self.remove_duplicate_sections = self.get_config(
            "remove_duplicate_sections",
            True,
        )

        self.remove_content_empty_sections = self.get_config(
            "remove_content_empty_sections",
            True,
        )

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Supprime les doublons présents dans le document.
        """

        statistics = {
            "paragraphs": 0,
            "list_items": 0,
            "table_rows": 0,
            "code_blocks": 0,
            "links": 0,
            "sections": 0,
        }

        global_seen: dict[str, set[str]] = {
            "paragraphs": set(),
            "list_items": set(),
            "table_rows": set(),
            "code_blocks": set(),
            "links": set(),
            "sections": set(),
        }

        cleaned_sections = []

        for section in document.sections:
            local_seen: dict[str, set[str]] = {
                "paragraphs": set(),
                "list_items": set(),
                "table_rows": set(),
                "code_blocks": set(),
                "links": set(),
            }

            # 1. Paragraphes
            if self.remove_duplicate_paragraphs:
                section.paragraphs = self._deduplicate_strings(
                    values=section.paragraphs,
                    local_seen=local_seen["paragraphs"],
                    global_seen=global_seen["paragraphs"],
                    statistics=statistics,
                    statistic_key="paragraphs",
                )

            # 2. Éléments de listes
            if self.remove_duplicate_list_items:
                cleaned_lists: list[list[str]] = []

                for items in section.lists:
                    cleaned_items = self._deduplicate_strings(
                        values=items,
                        local_seen=local_seen["list_items"],
                        global_seen=global_seen["list_items"],
                        statistics=statistics,
                        statistic_key="list_items",
                    )

                    if cleaned_items:
                        cleaned_lists.append(cleaned_items)

                section.lists = cleaned_lists

            # 3. Lignes de tableaux
            if self.remove_duplicate_table_rows:
                for table in section.tables:
                    cleaned_rows: list[list[str]] = []

                    for row in table.rows:
                        fingerprint = self._fingerprint_sequence(row)

                        if self._already_seen(
                            fingerprint=fingerprint,
                            local_seen=local_seen["table_rows"],
                            global_seen=global_seen["table_rows"],
                        ):
                            statistics["table_rows"] += 1
                            continue

                        cleaned_rows.append(row)

                    table.rows = cleaned_rows

            # 4. Blocs de code
            if self.remove_duplicate_code_blocks:
                cleaned_code_blocks = []

                for code_block in section.code_blocks:
                    code_value = (
                        f"{code_block.language or ''}\n"
                        f"{code_block.content or ''}"
                    )

                    fingerprint = self._fingerprint_text(
                        code_value
                    )

                    if self._already_seen(
                        fingerprint=fingerprint,
                        local_seen=local_seen["code_blocks"],
                        global_seen=global_seen["code_blocks"],
                    ):
                        statistics["code_blocks"] += 1
                        continue

                    cleaned_code_blocks.append(code_block)

                section.code_blocks = cleaned_code_blocks

            # 5. Liens
            if self.remove_duplicate_links:
                cleaned_links = []

                for link in section.links:
                    link_value = (
                        f"{link.text or ''}\n"
                        f"{link.url or ''}"
                    )

                    fingerprint = self._fingerprint_text(
                        link_value
                    )

                    if self._already_seen(
                        fingerprint=fingerprint,
                        local_seen=local_seen["links"],
                        global_seen=global_seen["links"],
                    ):
                        statistics["links"] += 1
                        continue

                    cleaned_links.append(link)

                section.links = cleaned_links

            # 6. Suppression des sections devenues vides
            if (
                self.remove_content_empty_sections
                and self._section_has_no_content(section)
            ):
                statistics["sections"] += 1
                continue

            # 7. Suppression des sections entièrement identiques
            if self.remove_duplicate_sections:
                section_fingerprint = (
                    self._section_fingerprint(section)
                )

                if (
                    section_fingerprint
                    in global_seen["sections"]
                ):
                    statistics["sections"] += 1
                    continue

                global_seen["sections"].add(
                    section_fingerprint
                )

            cleaned_sections.append(section)

        document.sections = cleaned_sections

        if document.metadata is None:
            document.metadata = {}

        document.metadata[
            "duplicates_removed"
        ] = statistics

        document.metadata[
            "duplicate_cleaner_config"
        ] = {
            "case_sensitive": self.case_sensitive,
            "global_scope": self.global_scope,
            "remove_content_empty_sections": (
                self.remove_content_empty_sections
            ),
        }

        return document

    def _deduplicate_strings(
        self,
        values: list[str],
        local_seen: set[str],
        global_seen: set[str],
        statistics: dict[str, int],
        statistic_key: str,
    ) -> list[str]:
        """
        Supprime les chaînes dupliquées en conservant
        l'ordre de la première occurrence.
        """

        cleaned_values: list[str] = []

        for value in values:
            fingerprint = self._fingerprint_text(value)

            if self._already_seen(
                fingerprint=fingerprint,
                local_seen=local_seen,
                global_seen=global_seen,
            ):
                statistics[statistic_key] += 1
                continue

            cleaned_values.append(value)

        return cleaned_values

    def _already_seen(
        self,
        fingerprint: str,
        local_seen: set[str],
        global_seen: set[str],
    ) -> bool:
        """
        Vérifie si une empreinte a déjà été rencontrée.

        global_scope=True :
        la comparaison est réalisée dans tout le document.

        global_scope=False :
        la comparaison est limitée à la section actuelle.
        """

        if not fingerprint:
            return False

        if fingerprint in local_seen:
            return True

        if (
            self.global_scope
            and fingerprint in global_seen
        ):
            return True

        local_seen.add(fingerprint)

        if self.global_scope:
            global_seen.add(fingerprint)

        return False

    def _fingerprint_text(
        self,
        value: str | None,
    ) -> str:
        """
        Crée une empreinte SHA-256 d'un texte normalisé.
        """

        normalized = self._normalize_text(value)

        if not normalized:
            return ""

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    def _fingerprint_sequence(
        self,
        values: list[str],
    ) -> str:
        """
        Crée une empreinte pour une ligne de tableau.
        """

        normalized_values = [
            self._normalize_text(value)
            for value in values
        ]

        combined = "\u241f".join(
            normalized_values
        )

        return hashlib.sha256(
            combined.encode("utf-8")
        ).hexdigest()

    def _normalize_text(
        self,
        value: str | None,
    ) -> str:
        """
        Normalise un texte avant la comparaison.

        Exemple :

        'Azure   Reliability'
        et
        'azure reliability'

        sont considérés identiques lorsque
        case_sensitive=False.
        """

        if value is None:
            return ""

        normalized = str(value).strip()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        if not self.case_sensitive:
            normalized = normalized.lower()

        return normalized

    def _section_fingerprint(
        self,
        section,
    ) -> str:
        """
        Crée une empreinte représentant le contenu
        complet d'une section.
        """

        section_content: list[str] = []

        section_content.append(
            self._normalize_text(section.heading)
        )

        section_content.extend(
            self._normalize_text(paragraph)
            for paragraph in section.paragraphs
        )

        for items in section.lists:
            section_content.extend(
                self._normalize_text(item)
                for item in items
            )

        for table in section.tables:
            section_content.extend(
                self._normalize_text(header)
                for header in table.headers
            )

            for row in table.rows:
                section_content.extend(
                    self._normalize_text(cell)
                    for cell in row
                )

        for code_block in section.code_blocks:
            section_content.append(
                self._normalize_text(
                    code_block.language
                )
            )

            section_content.append(
                self._normalize_text(
                    code_block.content
                )
            )

        for link in section.links:
            section_content.append(
                self._normalize_text(link.text)
            )

            section_content.append(
                self._normalize_text(link.url)
            )

        combined = "\u241e".join(
            section_content
        )

        return hashlib.sha256(
            combined.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _section_has_no_content(
        section,
    ) -> bool:
        """
        Vérifie si une section ne contient plus aucun
        contenu utile.

        Un titre seul n'est pas considéré comme suffisant
        pour créer un futur chunk.
        """

        has_paragraphs = bool(
            section.paragraphs
        )

        has_lists = any(
            items
            for items in section.lists
        )

        has_tables = any(
            table.headers or table.rows
            for table in section.tables
        )

        has_code_blocks = any(
            code_block.content
            for code_block in section.code_blocks
        )

        has_links = any(
            link.text or link.url
            for link in section.links
        )

        return not any(
            [
                has_paragraphs,
                has_lists,
                has_tables,
                has_code_blocks,
                has_links,
            ]
        )