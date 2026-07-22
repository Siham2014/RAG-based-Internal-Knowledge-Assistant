from __future__ import annotations

import re

from src.cleaning.base import BaseCleaner
from src.cleaning.registry import CleanerRegistry
from src.models.document import ParsedDocument


@CleanerRegistry.register(
    "boilerplate",
    "boilerplate_cleaner",
)
class BoilerplateCleaner(BaseCleaner):
    """
    Supprime les contenus répétitifs et non informatifs.

    Exemples :
    - Edit
    - Feedback
    - Previous
    - Next
    - Table of contents
    - Sign in
    - Copyright Microsoft
    """

    cleaner_name = "boilerplate_cleaner"

    DEFAULT_EXACT_PATTERNS = {
        "edit",
        "edit this page",
        "edit on github",
        "feedback",
        "submit feedback",
        "previous",
        "next",
        "table of contents",
        "in this article",
        "sign in",
        "sign out",
        "print",
        "share",
        "learn",
        "skip to main content",
        "was this page helpful?",
        "is this page helpful?",
        "need help?",
        "additional resources",
        "related content",
        "related articles",
        "privacy",
        "terms of use",
        "trademarks",
    }

    DEFAULT_REGEX_PATTERNS = [
        r"^©\s*\d{4}.*$",
        r"^copyright\s+\d{4}.*$",
        r"^last updated[:\s].*$",
        r"^updated[:\s].*$",
        r"^page last reviewed[:\s].*$",
        r"^this page was last updated.*$",
        r"^contribute to this page.*$",
        r"^open a documentation issue.*$",
        r"^view all page feedback.*$",
    ]

    def __init__(
        self,
        enabled: bool = True,
        config: dict | None = None,
    ) -> None:
        super().__init__(
            enabled=enabled,
            config=config,
        )

        custom_exact_patterns = self.get_config(
            "exact_patterns",
            [],
        )

        custom_regex_patterns = self.get_config(
            "regex_patterns",
            [],
        )

        self.exact_patterns = {
            self._normalize_for_comparison(pattern)
            for pattern in (
                list(self.DEFAULT_EXACT_PATTERNS)
                + list(custom_exact_patterns)
            )
        }

        self.regex_patterns = [
            re.compile(
                pattern,
                flags=re.IGNORECASE,
            )
            for pattern in (
                list(self.DEFAULT_REGEX_PATTERNS)
                + list(custom_regex_patterns)
            )
        ]

        self.remove_empty_sections = self.get_config(
            "remove_empty_sections",
            True,
        )

        self.remove_boilerplate_links = self.get_config(
            "remove_boilerplate_links",
            True,
        )

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Supprime les contenus identifiés comme boilerplate.
        """

        cleaned_sections = []

        removed_paragraph_count = 0
        removed_list_item_count = 0
        removed_link_count = 0
        removed_section_count = 0

        for section in document.sections:
            if self._is_boilerplate(
                section.heading
            ):
                removed_section_count += 1
                continue

            cleaned_paragraphs = []

            for paragraph in section.paragraphs:
                if self._is_boilerplate(paragraph):
                    removed_paragraph_count += 1
                    continue

                cleaned_paragraphs.append(
                    paragraph
                )

            section.paragraphs = cleaned_paragraphs

            cleaned_lists: list[list[str]] = []

            for items in section.lists:
                cleaned_items = []

                for item in items:
                    if self._is_boilerplate(item):
                        removed_list_item_count += 1
                        continue

                    cleaned_items.append(item)

                if cleaned_items:
                    cleaned_lists.append(
                        cleaned_items
                    )

            section.lists = cleaned_lists

            if self.remove_boilerplate_links:
                cleaned_links = []

                for link in section.links:
                    if (
                        self._is_boilerplate(link.text)
                        or self._is_boilerplate_url(
                            link.url
                        )
                    ):
                        removed_link_count += 1
                        continue

                    cleaned_links.append(link)

                section.links = cleaned_links

            if (
                self.remove_empty_sections
                and self._section_is_empty(section)
            ):
                removed_section_count += 1
                continue

            cleaned_sections.append(section)

        document.sections = cleaned_sections

        if document.metadata is None:
            document.metadata = {}

        document.metadata[
            "boilerplate_removed"
        ] = {
            "paragraphs": removed_paragraph_count,
            "list_items": removed_list_item_count,
            "links": removed_link_count,
            "sections": removed_section_count,
        }

        return document

    def _is_boilerplate(
        self,
        value: str | None,
    ) -> bool:
        """
        Vérifie si un texte correspond à un contenu
        répétitif ou non informatif.
        """

        if value is None:
            return False

        normalized = self._normalize_for_comparison(
            value
        )

        if not normalized:
            return False

        if normalized in self.exact_patterns:
            return True

        for pattern in self.regex_patterns:
            if pattern.fullmatch(
                normalized
            ):
                return True

        return False

    @staticmethod
    def _normalize_for_comparison(
        value: str,
    ) -> str:
        """
        Normalise un texte avant comparaison.
        """

        normalized = str(value).strip().lower()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        normalized = normalized.strip(
            " \t\r\n:;,.!?-–—|"
        )

        return normalized

    @staticmethod
    def _is_boilerplate_url(
        url: str | None,
    ) -> bool:
        """
        Détecte certains liens inutiles pour la base RAG.
        """

        if not url:
            return False

        normalized_url = url.strip().lower()

        unwanted_fragments = (
            "/feedback",
            "github.com/microsoftdocs",
            "#feedback",
            "/legal/",
            "/privacy",
            "/terms",
        )

        return any(
            fragment in normalized_url
            for fragment in unwanted_fragments
        )

    @staticmethod
    def _section_is_empty(
        section,
    ) -> bool:
        """
        Vérifie si une section ne contient plus aucune
        information utile.
        """

        has_heading = bool(
            section.heading
            and section.heading.strip()
        )

        has_paragraphs = bool(
            section.paragraphs
        )

        has_lists = any(
            items
            for items in section.lists
        )

        has_tables = bool(
            section.tables
        )

        has_code = bool(
            section.code_blocks
        )

        has_links = bool(
            section.links
        )

        return not any(
            [
                has_heading,
                has_paragraphs,
                has_lists,
                has_tables,
                has_code,
                has_links,
            ]
        )