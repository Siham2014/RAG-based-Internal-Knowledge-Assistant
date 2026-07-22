from __future__ import annotations

from pathlib import Path
from typing import Any

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


@ParserRegistry.register(".html", ".htm")
class HtmlParser(BaseParser):
    """
    Parser intermédiaire pour les fichiers HTML.

    Il extrait :
    - le titre ;
    - les sections h1 à h6 ;
    - les paragraphes ;
    - les listes ;
    - les tableaux ;
    - les blocs de code ;
    - les liens ;
    - les métadonnées HTML principales.
    """

    def parse(
        self,
        file_path: Path,
    ) -> ParsedDocument:
        file_path = file_path.resolve()

        self.validate_file(file_path)

        html_content = self._read_file(file_path)

        soup = BeautifulSoup(
            html_content,
            "html.parser",
        )

        self._remove_useless_elements(soup)

        document_id = self.build_document_id(
            file_path
        )

        content_root = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        sections = self._extract_sections(
            root=content_root,
            document_id=document_id,
        )

        html_metadata = self._extract_html_metadata(
            soup
        )

        source_metadata = self._get_source_metadata()

        title = self._extract_title(
            soup=soup,
            sections=sections,
            file_path=file_path,
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
                html_metadata.get("author")
                or source_metadata.get("author")
            ),
            source_url=(
                html_metadata.get("canonical_url")
                or self._get_source_url()
            ),
            language=(
                html_metadata.get("language")
                or source_metadata.get("language")
            ),
            organization=source_metadata.get(
                "organization"
            ),
            document_format="html",
            sections=sections,
            metadata={
                **source_metadata,
                **html_metadata,
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
        Lit un fichier HTML.

        Une seconde tentative est réalisée avec Windows-1252
        lorsque le fichier n'est pas encodé en UTF-8.
        """

        try:
            return file_path.read_text(
                encoding="utf-8-sig"
            )

        except UnicodeDecodeError:
            try:
                return file_path.read_text(
                    encoding="windows-1252"
                )

            except UnicodeDecodeError as error:
                raise ParserError(
                    "Impossible de décoder le fichier HTML : "
                    f"{file_path}"
                ) from error

        except OSError as error:
            raise ParserError(
                f"Erreur de lecture du fichier {file_path}: "
                f"{error}"
            ) from error

    @staticmethod
    def _remove_useless_elements(
        soup: BeautifulSoup,
    ) -> None:
        """
        Supprime les éléments qui ne doivent pas entrer
        dans le contenu documentaire.
        """

        for element in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "template",
                "svg",
                "canvas",
            ]
        ):
            element.decompose()

    def _extract_sections(
        self,
        root: Tag | BeautifulSoup,
        document_id: str,
    ) -> list[DocumentSection]:
        """
        Découpe le contenu selon les titres h1 à h6.
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

        elements = root.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "table",
                "pre",
                "blockquote",
            ]
        )

        for element in elements:
            if self._has_supported_parent(
                element=element,
                root=root,
            ):
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
                        title
                        for _, title in heading_stack
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
            section.section_id = (
                f"{document_id}:section-{position}"
            )

        return sections

    @staticmethod
    def _has_supported_parent(
        element: Tag,
        root: Tag | BeautifulSoup,
    ) -> bool:
        """
        Empêche le double traitement des éléments.

        Exemple :
        un paragraphe placé dans un blockquote ne doit pas
        être traité une fois comme paragraphe puis une seconde
        fois comme blockquote.
        """

        supported_names = {
            "p",
            "ul",
            "ol",
            "table",
            "pre",
            "blockquote",
        }

        parent = element.parent

        while isinstance(parent, Tag) and parent is not root:
            if parent.name in supported_names:
                return True

            parent = parent.parent

        return False

    @staticmethod
    def _create_section(
        document_id: str,
        position: int,
        heading: str | None,
        heading_level: int | None,
        heading_path: list[str],
    ) -> DocumentSection:
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

        self._add_links_from_element(
            element=element,
            section=section,
        )

    @staticmethod
    def _add_paragraph(
        element: Tag,
        section: DocumentSection,
    ) -> None:
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
        headers: list[str] = []
        rows: list[list[str]] = []

        thead = element.find("thead")

        if thead is not None:
            first_header_row = thead.find("tr")

            if first_header_row is not None:
                headers = [
                    cell.get_text(" ", strip=True)
                    for cell in first_header_row.find_all(
                        ["th", "td"],
                        recursive=False,
                    )
                ]

        tbody = element.find("tbody")

        rows_container = (
            tbody if tbody is not None else element
        )

        for row in rows_container.find_all(
            "tr",
            recursive=False,
        ):
            cells_tags = row.find_all(
                ["th", "td"],
                recursive=False,
            )

            cells = [
                cell.get_text(" ", strip=True)
                for cell in cells_tags
            ]

            if not cells:
                continue

            if (
                not headers
                and cells_tags
                and all(
                    cell.name == "th"
                    for cell in cells_tags
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

            classes = code_element.get(
                "class",
                [],
            )

            for css_class in classes:
                if css_class.startswith("language-"):
                    language = css_class.replace(
                        "language-",
                        "",
                        1,
                    )
                    break

                if css_class.startswith("lang-"):
                    language = css_class.replace(
                        "lang-",
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
        text = element.get_text(
            " ",
            strip=True,
        )

        if text:
            section.paragraphs.append(
                f"> {text}"
            )

    @staticmethod
    def _add_links_from_element(
        element: Tag,
        section: DocumentSection,
    ) -> None:
        existing_links = {
            (link.text, link.url)
            for link in section.links
        }

        link_elements = []

        if element.name == "a":
            link_elements.append(element)

        link_elements.extend(
            element.find_all("a")
        )

        for link_element in link_elements:
            url = str(
                link_element.get("href", "")
            ).strip()

            text = link_element.get_text(
                " ",
                strip=True,
            )

            if not url:
                continue

            key = (text, url)

            if key in existing_links:
                continue

            section.links.append(
                DocumentLink(
                    text=text,
                    url=url,
                )
            )

            existing_links.add(key)

    @staticmethod
    def _is_heading(
        element: Tag,
    ) -> bool:
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
        soup: BeautifulSoup,
        sections: list[DocumentSection],
        file_path: Path,
    ) -> str:
        """
        Priorité :
        1. premier titre H1 ;
        2. balise HTML title ;
        3. premier titre disponible ;
        4. nom du fichier.
        """

        for section in sections:
            if (
                section.heading
                and section.heading_level == 1
            ):
                return section.heading

        if soup.title:
            title_text = soup.title.get_text(
                " ",
                strip=True,
            )

            if title_text:
                return title_text

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

    @staticmethod
    def _extract_html_metadata(
        soup: BeautifulSoup,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        html_tag = soup.find("html")

        if html_tag is not None:
            language = html_tag.get("lang")

            if language:
                metadata["language"] = str(
                    language
                )

        description = soup.find(
            "meta",
            attrs={"name": "description"},
        )

        if description is not None:
            content = description.get("content")

            if content:
                metadata["description"] = str(
                    content
                )

        author = soup.find(
            "meta",
            attrs={"name": "author"},
        )

        if author is not None:
            content = author.get("content")

            if content:
                metadata["author"] = str(
                    content
                )

        keywords = soup.find(
            "meta",
            attrs={"name": "keywords"},
        )

        if keywords is not None:
            content = keywords.get("content")

            if content:
                metadata["keywords"] = [
                    value.strip()
                    for value in str(content).split(",")
                    if value.strip()
                ]

        canonical = soup.find(
            "link",
            attrs={"rel": "canonical"},
        )

        if canonical is not None:
            href = canonical.get("href")

            if href:
                metadata["canonical_url"] = str(
                    href
                )

        return metadata

    def _get_source_metadata(
        self,
    ) -> dict[str, Any]:
        metadata = getattr(
            self.source,
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            return dict(metadata)

        return {}

    def _get_source_url(self) -> str | None:
        input_config = getattr(
            self.source,
            "input",
            {},
        )

        if not isinstance(input_config, dict):
            return None

        url = (
            input_config.get("url")
            or input_config.get("base_url")
        )

        return str(url) if url else None