from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.chunking.models import TextBlock


class BlockBuilderError(RuntimeError):
    """
    Erreur déclenchée pendant la création des blocs documentaires.
    """


class BlockBuilder:
    """
    Transforme un document parsé et nettoyé en liste de TextBlock.

    Cette première version accepte :
    - un dictionnaire Python ;
    - un objet possédant des attributs ;
    - une liste de sections ;
    - une liste de blocs déjà structurés.

    Elle évite de coupler immédiatement le chunking à une seule
    implémentation de ParsedDocument.
    """

    SUPPORTED_BLOCK_TYPES = {
        "title",
        "heading",
        "paragraph",
        "list",
        "table",
        "code",
        "quote",
        "text",
    }

    TYPE_ALIASES = {
        "header": "heading",
        "section_title": "heading",
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "h5": "heading",
        "h6": "heading",
        "unordered_list": "list",
        "ordered_list": "list",
        "bullet_list": "list",
        "code_block": "code",
        "preformatted": "code",
        "body": "paragraph",
    }

    def build(self, document: Any) -> list[TextBlock]:
        """
        Construit les TextBlock d'un document.
        """

        if document is None:
            raise BlockBuilderError(
                "Le document ne peut pas être None."
            )

        document_id = self._read_required_value(
            document,
            [
                "document_id",
                "id",
                "doc_id",
            ],
        )

        document_title = self._read_optional_value(
            document,
            [
                "document_title",
                "title",
                "name",
            ],
        )

        source_url = self._read_optional_value(
            document,
            [
                "source_url",
                "url",
                "source",
            ],
        )

        language = self._read_optional_value(
            document,
            [
                "language",
                "lang",
            ],
        )

        sections = self._read_optional_value(
            document,
            [
                "sections",
                "content",
                "blocks",
            ],
        )

        if sections is None:
            raw_text = self._read_optional_value(
                document,
                [
                    "text",
                    "cleaned_text",
                    "body",
                ],
            )

            if raw_text is None:
                raise BlockBuilderError(
                    "Aucun contenu exploitable n'a été trouvé dans le document."
                )

            sections = [
                {
                    "type": "paragraph",
                    "text": str(raw_text),
                }
            ]

        blocks: list[TextBlock] = []

        self._append_items(
            items=sections,
            blocks=blocks,
            document_id=str(document_id),
            document_title=self._to_optional_string(document_title),
            source_url=self._to_optional_string(source_url),
            language=self._to_optional_string(language),
            inherited_section_id=None,
            inherited_section_heading=None,
            inherited_heading_path=[],
        )

        if not blocks:
            raise BlockBuilderError(
                "Le document n'a produit aucun bloc textuel."
            )

        return blocks

    def _append_items(
        self,
        *,
        items: Any,
        blocks: list[TextBlock],
        document_id: str,
        document_title: str | None,
        source_url: str | None,
        language: str | None,
        inherited_section_id: str | None,
        inherited_section_heading: str | None,
        inherited_heading_path: list[str],
    ) -> None:
        if isinstance(items, str):
            self._append_block(
                item={
                    "type": "paragraph",
                    "text": items,
                },
                blocks=blocks,
                document_id=document_id,
                document_title=document_title,
                source_url=source_url,
                language=language,
                inherited_section_id=inherited_section_id,
                inherited_section_heading=inherited_section_heading,
                inherited_heading_path=inherited_heading_path,
            )
            return

        if isinstance(items, Mapping):
            iterable_items: Iterable[Any] = [items]
        elif isinstance(items, Iterable):
            iterable_items = items
        else:
            iterable_items = [items]

        for item in iterable_items:
            if item is None:
                continue

            children = self._read_optional_value(
                item,
                [
                    "children",
                    "content",
                    "blocks",
                    "elements",
                ],
            )

            item_type = self._normalize_block_type(
                self._read_optional_value(
                    item,
                    [
                        "block_type",
                        "type",
                        "element_type",
                        "kind",
                    ],
                )
            )

            section_id = self._to_optional_string(
                self._read_optional_value(
                    item,
                    [
                        "section_id",
                        "id",
                    ],
                )
            ) or inherited_section_id

            section_heading = self._to_optional_string(
                self._read_optional_value(
                    item,
                    [
                        "section_heading",
                        "heading",
                        "title",
                    ],
                )
            ) or inherited_section_heading

            heading_path = self._extract_heading_path(
                item=item,
                inherited=inherited_heading_path,
                section_heading=section_heading,
                item_type=item_type,
            )

            text = self._extract_text(item)

            if text:
                self._append_block(
                    item=item,
                    blocks=blocks,
                    document_id=document_id,
                    document_title=document_title,
                    source_url=source_url,
                    language=language,
                    inherited_section_id=section_id,
                    inherited_section_heading=section_heading,
                    inherited_heading_path=heading_path,
                )

            if children is not None and children is not item:
                self._append_items(
                    items=children,
                    blocks=blocks,
                    document_id=document_id,
                    document_title=document_title,
                    source_url=source_url,
                    language=language,
                    inherited_section_id=section_id,
                    inherited_section_heading=section_heading,
                    inherited_heading_path=heading_path,
                )

    def _append_block(
        self,
        *,
        item: Any,
        blocks: list[TextBlock],
        document_id: str,
        document_title: str | None,
        source_url: str | None,
        language: str | None,
        inherited_section_id: str | None,
        inherited_section_heading: str | None,
        inherited_heading_path: list[str],
    ) -> None:
        text = self._extract_text(item)

        if not text:
            return

        block_type = self._normalize_block_type(
            self._read_optional_value(
                item,
                [
                    "block_type",
                    "type",
                    "element_type",
                    "kind",
                ],
            )
        )

        metadata = self._read_optional_value(
            item,
            [
                "metadata",
                "meta",
            ],
        )

        if not isinstance(metadata, Mapping):
            metadata = {}

        start_position = self._read_optional_value(
            item,
            [
                "start_position",
                "start",
                "char_start",
            ],
        )

        end_position = self._read_optional_value(
            item,
            [
                "end_position",
                "end",
                "char_end",
            ],
        )

        block = TextBlock.create(
            document_id=document_id,
            block_type=block_type,
            text=text,
            block_index=len(blocks),
            section_id=inherited_section_id,
            section_heading=inherited_section_heading,
            heading_path=inherited_heading_path,
            source_url=source_url,
            document_title=document_title,
            language=language,
            start_position=self._to_optional_integer(start_position),
            end_position=self._to_optional_integer(end_position),
            metadata=dict(metadata),
        )

        blocks.append(block)

    def _extract_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item.strip()

        value = self._read_optional_value(
            item,
            [
                "text",
                "content_text",
                "value",
                "raw_text",
            ],
        )

        if value is None:
            return ""

        if isinstance(value, list):
            return "\n".join(
                str(element).strip()
                for element in value
                if str(element).strip()
            )

        return str(value).strip()

    def _normalize_block_type(self, value: Any) -> str:
        if value is None:
            return "text"

        normalized = str(value).strip().lower()
        normalized = self.TYPE_ALIASES.get(
            normalized,
            normalized,
        )

        if normalized not in self.SUPPORTED_BLOCK_TYPES:
            return "text"

        return normalized

    def _extract_heading_path(
        self,
        *,
        item: Any,
        inherited: list[str],
        section_heading: str | None,
        item_type: str,
    ) -> list[str]:
        explicit_path = self._read_optional_value(
            item,
            [
                "heading_path",
                "breadcrumb",
                "breadcrumbs",
            ],
        )

        if explicit_path:
            if isinstance(explicit_path, str):
                return [
                    part.strip()
                    for part in explicit_path.split(">")
                    if part.strip()
                ]

            if isinstance(explicit_path, Iterable):
                return [
                    str(part).strip()
                    for part in explicit_path
                    if str(part).strip()
                ]

        path = list(inherited)

        if (
            item_type in {"title", "heading"}
            and section_heading
            and (
                not path
                or path[-1] != section_heading
            )
        ):
            path.append(section_heading)

        return path

    @staticmethod
    def _read_optional_value(
        source: Any,
        names: list[str],
    ) -> Any:
        if isinstance(source, Mapping):
            for name in names:
                if name in source:
                    return source[name]

            return None

        for name in names:
            if hasattr(source, name):
                return getattr(source, name)

        return None

    def _read_required_value(
        self,
        source: Any,
        names: list[str],
    ) -> Any:
        value = self._read_optional_value(
            source,
            names,
        )

        if value is None or not str(value).strip():
            joined_names = ", ".join(names)

            raise BlockBuilderError(
                f"Champ obligatoire absent. "
                f"Champs recherchés : {joined_names}."
            )

        return value

    @staticmethod
    def _to_optional_string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        result = str(value).strip()

        return result or None

    @staticmethod
    def _to_optional_integer(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise BlockBuilderError(
                f"Position invalide : {value!r}."
            ) from exc