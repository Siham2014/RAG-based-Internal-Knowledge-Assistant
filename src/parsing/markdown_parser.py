from __future__ import annotations

from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup, Tag

from src.models.document import (
    CodeBlock,
    DocumentLink,
    DocumentSection,
    DocumentTable,
    ParsedDocument,
)
from src.parsing.base import BaseParser, ParserError
from src.parsing.registry import ParserRegistry


@ParserRegistry.register(".md", ".markdown")
class MarkdownParser(BaseParser):
    """
    Parser intermédiaire pour les fichiers Markdown.

    Étapes :
    1. Lecture du fichier Markdown.
    2. Conversion Markdown vers HTML.
    3. Analyse du HTML avec BeautifulSoup.
    4. Construction d'un ParsedDocument structuré.
    """

    markdown_extensions = [
        "fenced_code",
        "tables",
        "sane_lists",
    ]

    def parse(
        self,
        file_path: Path,
    ) -> ParsedDocument:
        """
        Parse un fichier Markdown et retourne un ParsedDocument.
        """

        file_path = file_path.resolve()

        self.validate_file(file_path)

        markdown_content = self._read_file(file_path)

        html_content = markdown.markdown(
            markdown_content,
            extensions=self.markdown_extensions,
            output_format="html5",
        )

        soup = BeautifulSoup(
            html_content,
            "html.parser",
        )

        document_id = self.build_document_id(
            file_path
        )

        sections = self._extract_sections(
            soup=soup,
            document_id=document_id,
        )

        title = self._extract_title(
            sections=sections,
            file_path=file_path,
        )

        source_metadata = self._get_source_metadata()

        return ParsedDocument(
            document_id=document_id,
            source_id=self.source.id,
            source_type=self.source.type,
            local_path=str(file_path),
            checksum_sha256=self.compute_sha256(
                file_path
            ),
            title=title,
            author=source_metadata.get("author"),
            source_url=self._get_source_url(),
            language=source_metadata.get("language"),
            organization=source_metadata.get(
                "organization"
            ),
            document_format="markdown",
            sections=sections,
            metadata={
                **source_metadata,
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "parser": self.__class__.__name__,
                "section_count": len(sections),
            },
        )

    @staticmethod
    def _read_file(
        file_path: Path,
    ) -> str:
        """
        Lit le fichier Markdown en UTF-8.

        utf-8-sig permet aussi de gérer certains fichiers
        contenant un BOM UTF-8.
        """

        try:
            return file_path.read_text(
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError as error:
            raise ParserError(
                "Impossible de lire le fichier Markdown "
                f"en UTF-8 : {file_path}"
            ) from error

        except OSError as error:
            raise ParserError(
                f"Erreur pendant la lecture de {file_path} : "
                f"{error}"
            ) from error

    def _extract_sections(
        self,
        soup: BeautifulSoup,
        document_id: str,
    ) -> list[DocumentSection]:
        """
        Découpe le document selon les titres HTML h1 à h6.

        Tout contenu placé avant le premier titre est enregistré
        dans une section d'introduction.
        """

        sections: list[DocumentSection] = []

        heading_stack: list[tuple[int, str]] = []

        current_section = self._create_section(
            document_id=document_id,
            position=0,
            heading=None,
            heading_level=None,
            heading_path=[],
        )

        for element in soup.children:
            if not isinstance(element, Tag):
                continue

            if self._is_heading(element):
                if not current_section.is_empty:
                    sections.append(current_section)

                heading_level = int(
                    element.name[1]
                )

                heading_text = element.get_text(
                    " ",
                    strip=True,
                )

                heading_stack = self._update_heading_stack(
                    heading_stack=heading_stack,
                    heading_level=heading_level,
                    heading_text=heading_text,
                )

                current_section = self._create_section(
                    document_id=document_id,
                    position=len(sections),
                    heading=heading_text,
                    heading_level=heading_level,
                    heading_path=[
                        heading
                        for _, heading in heading_stack
                    ],
                )

                self._add_links_from_element(
                    element=element,
                    section=current_section,
                )

                continue

            self._process_element(
                element=element,
                section=current_section,
            )

        if not current_section.is_empty:
            sections.append(current_section)

        for position, section in enumerate(sections):
            section.position = position

        return sections

    @staticmethod
    def _create_section(
        document_id: str,
        position: int,
        heading: str | None,
        heading_level: int | None,
        heading_path: list[str],
    ) -> DocumentSection:
        """
        Crée une section avec un identifiant stable.
        """

        return DocumentSection(
            section_id=(
                f"{document_id}:section-{position}"
            ),
            heading=heading,
            heading_level=heading_level,
            position=position,
            heading_path=heading_path,
        )

    def _process_element(
        self,
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Analyse un élément HTML de premier niveau.
        """

        element_name = element.name.lower()

        if element_name == "p":
            self._add_paragraph(
                element=element,
                section=section,
            )

        elif element_name in {"ul", "ol"}:
            self._add_list(
                element=element,
                section=section,
            )

        elif element_name == "table":
            table = self._extract_table(element)

            if table is not None:
                section.tables.append(table)

        elif element_name == "pre":
            self._add_code_block(
                element=element,
                section=section,
            )

        elif element_name == "blockquote":
            self._add_blockquote(
                element=element,
                section=section,
            )

        elif element_name == "hr":
            return

        else:
            self._process_nested_content(
                element=element,
                section=section,
            )

        self._add_links_from_element(
            element=element,
            section=section,
        )

    @staticmethod
    def _add_paragraph(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Ajoute un paragraphe à la section.
        """

        text = element.get_text(
            " ",
            strip=True,
        )

        if text:
            section.paragraphs.append(text)

    @staticmethod
    def _add_list(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Extrait les éléments directs d'une liste.

        recursive=False évite de mélanger les éléments
        des sous-listes dans la liste parente.
        """

        items: list[str] = []

        for list_item in element.find_all(
            "li",
            recursive=False,
        ):
            text = list_item.get_text(
                " ",
                strip=True,
            )

            if text:
                items.append(text)

        if items:
            section.lists.append(items)

    @staticmethod
    def _extract_table(
        element: Tag,
    ) -> DocumentTable | None:
        """
        Extrait un tableau HTML produit depuis le Markdown.
        """

        headers: list[str] = []
        rows: list[list[str]] = []

        header_row = element.find("thead")

        if header_row is not None:
            headers = [
                cell.get_text(" ", strip=True)
                for cell in header_row.find_all(
                    ["th", "td"]
                )
            ]

        body = element.find("tbody")

        row_container = (
            body if body is not None else element
        )

        for row in row_container.find_all(
            "tr",
            recursive=False,
        ):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(
                    ["td", "th"],
                    recursive=False,
                )
            ]

            if not cells:
                continue

            if not headers and all(
                cell.name == "th"
                for cell in row.find_all(
                    ["td", "th"],
                    recursive=False,
                )
            ):
                headers = cells
                continue

            rows.append(cells)

        if not headers and not rows:
            return None

        return DocumentTable(
            headers=headers,
            rows=rows,
        )

    @staticmethod
    def _add_code_block(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Extrait un bloc de code et son langage éventuel.

        Exemple HTML :

        <pre>
            <code class="language-python">
                ...
            </code>
        </pre>
        """

        code_element = element.find("code")

        if code_element is None:
            content = element.get_text(
                "\n",
                strip=False,
            )

            language = None

        else:
            content = code_element.get_text(
                "",
                strip=False,
            )

            language = None

            for css_class in code_element.get(
                "class",
                [],
            ):
                if css_class.startswith("language-"):
                    language = css_class.replace(
                        "language-",
                        "",
                        1,
                    )

                    break

        cleaned_content = content.strip("\n")

        if cleaned_content:
            section.code_blocks.append(
                CodeBlock(
                    content=cleaned_content,
                    language=language,
                )
            )

    @staticmethod
    def _add_blockquote(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Conserve le contenu des citations Markdown.
        """

        text = element.get_text(
            " ",
            strip=True,
        )

        if text:
            section.paragraphs.append(
                f"> {text}"
            )

    def _process_nested_content(
        self,
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Traite les balises conteneurs qui peuvent contenir
        plusieurs éléments importants.
        """

        important_children = element.find_all(
            [
                "p",
                "ul",
                "ol",
                "table",
                "pre",
                "blockquote",
            ],
            recursive=False,
        )

        if not important_children:
            text = element.get_text(
                " ",
                strip=True,
            )

            if text:
                section.paragraphs.append(text)

            return

        for child in important_children:
            self._process_element(
                element=child,
                section=section,
            )

    @staticmethod
    def _add_links_from_element(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        """
        Ajoute les liens trouvés dans un élément.

        Les doublons sont ignorés.
        """

        existing_links = {
            (link.text, link.url)
            for link in section.links
        }

        links = []

        if element.name == "a":
            links.append(element)

        links.extend(
            element.find_all("a")
        )

        for link_element in links:
            url = str(
                link_element.get("href", "")
            ).strip()

            text = link_element.get_text(
                " ",
                strip=True,
            )

            if not url:
                continue

            link_key = (text, url)

            if link_key in existing_links:
                continue

            section.links.append(
                DocumentLink(
                    text=text,
                    url=url,
                )
            )

            existing_links.add(link_key)

    @staticmethod
    def _is_heading(
        element: Tag,
    ) -> bool:
        """
        Vérifie si la balise est un titre h1 à h6.
        """

        return element.name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }

    @staticmethod
    def _update_heading_stack(
        heading_stack: list[tuple[int, str]],
        heading_level: int,
        heading_text: str,
    ) -> list[tuple[int, str]]:
        """
        Met à jour le chemin hiérarchique des titres.

        Exemple :

        H1 Azure
        H2 Reliability
        H3 Retry

        donne :

        ["Azure", "Reliability", "Retry"]
        """

        updated_stack = [
            (level, title)
            for level, title in heading_stack
            if level < heading_level
        ]

        updated_stack.append(
            (
                heading_level,
                heading_text,
            )
        )

        return updated_stack

    @staticmethod
    def _extract_title(
        sections: list[DocumentSection],
        file_path: Path,
    ) -> str:
        """
        Utilise le premier titre H1 comme titre du document.

        Si aucun H1 n'existe, le nom du fichier est utilisé.
        """

        for section in sections:
            if (
                section.heading
                and section.heading_level == 1
            ):
                return section.heading

        for section in sections:
            if section.heading:
                return section.heading

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

    def _get_source_url(self) -> str | None:
        """
        Récupère une URL éventuelle dans la configuration source.
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
            or input_config.get("repository_url")
        )

        return str(url) if url else None