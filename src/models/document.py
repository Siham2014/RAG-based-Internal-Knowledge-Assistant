from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentLink:
    """
    Représente un lien détecté dans une section.
    """

    text: str
    url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DocumentLink":
        return cls(
            text=str(data.get("text", "")),
            url=str(data.get("url", "")),
        )


@dataclass
class DocumentTable:
    """
    Représente un tableau extrait d'un document.

    headers :
        noms des colonnes.

    rows :
        contenu des lignes.
    """

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DocumentTable":
        return cls(
            headers=[
                str(value)
                for value in data.get("headers", [])
            ],
            rows=[
                [
                    str(value)
                    for value in row
                ]
                for row in data.get("rows", [])
            ],
            caption=data.get("caption"),
        )


@dataclass
class CodeBlock:
    """
    Représente un bloc de code extrait.

    language peut contenir :
    python, yaml, bash, json, powershell, etc.
    """

    content: str
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "CodeBlock":
        return cls(
            content=str(data.get("content", "")),
            language=data.get("language"),
        )


@dataclass
class DocumentSection:
    """
    Représente une section structurée d'un document.

    Une section peut contenir :
    - un titre ;
    - des paragraphes ;
    - des listes ;
    - des tableaux ;
    - des blocs de code ;
    - des liens ;
    - un numéro de page pour les PDF.
    """

    section_id: str
    heading: str | None = None
    heading_level: int | None = None

    paragraphs: list[str] = field(default_factory=list)
    lists: list[list[str]] = field(default_factory=list)
    tables: list[DocumentTable] = field(default_factory=list)
    code_blocks: list[CodeBlock] = field(default_factory=list)
    links: list[DocumentLink] = field(default_factory=list)

    page_number: int | None = None
    position: int = 0

    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise ValueError(
                "section_id ne peut pas être vide."
            )

        if (
            self.heading_level is not None
            and self.heading_level < 1
        ):
            raise ValueError(
                "heading_level doit être supérieur ou égal à 1."
            )

        if self.page_number is not None and self.page_number < 1:
            raise ValueError(
                "page_number doit être supérieur ou égal à 1."
            )

        if self.position < 0:
            raise ValueError(
                "position ne peut pas être négative."
            )

    @property
    def text(self) -> str:
        """
        Reconstruit une version textuelle de la section.

        Cette propriété sera utile pour :
        - le nettoyage ;
        - le chunking ;
        - l'indexation ;
        - l'évaluation.
        """

        parts: list[str] = []

        if self.heading:
            parts.append(self.heading.strip())

        parts.extend(
            paragraph.strip()
            for paragraph in self.paragraphs
            if paragraph.strip()
        )

        for item_list in self.lists:
            cleaned_items = [
                item.strip()
                for item in item_list
                if item.strip()
            ]

            if cleaned_items:
                parts.append(
                    "\n".join(
                        f"- {item}"
                        for item in cleaned_items
                    )
                )

        for table in self.tables:
            table_lines: list[str] = []

            if table.caption:
                table_lines.append(table.caption.strip())

            if table.headers:
                table_lines.append(
                    " | ".join(table.headers)
                )

            for row in table.rows:
                table_lines.append(
                    " | ".join(row)
                )

            if table_lines:
                parts.append("\n".join(table_lines))

        for code_block in self.code_blocks:
            language = code_block.language or ""

            parts.append(
                f"```{language}\n"
                f"{code_block.content.strip()}\n"
                "```"
            )

        return "\n\n".join(
            part
            for part in parts
            if part.strip()
        )

    @property
    def is_empty(self) -> bool:
        """
        Indique si la section ne contient aucun contenu utile.
        """

        return not bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit la section en dictionnaire JSON-compatible.
        """

        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "paragraphs": self.paragraphs,
            "lists": self.lists,
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "code_blocks": [
                code_block.to_dict()
                for code_block in self.code_blocks
            ],
            "links": [
                link.to_dict()
                for link in self.links
            ],
            "page_number": self.page_number,
            "position": self.position,
            "heading_path": self.heading_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DocumentSection":
        """
        Reconstruit une section depuis un dictionnaire.
        """

        return cls(
            section_id=str(data["section_id"]),
            heading=data.get("heading"),
            heading_level=data.get("heading_level"),
            paragraphs=[
                str(paragraph)
                for paragraph in data.get(
                    "paragraphs",
                    [],
                )
            ],
            lists=[
                [
                    str(item)
                    for item in item_list
                ]
                for item_list in data.get(
                    "lists",
                    [],
                )
            ],
            tables=[
                DocumentTable.from_dict(table)
                for table in data.get(
                    "tables",
                    [],
                )
            ],
            code_blocks=[
                CodeBlock.from_dict(code_block)
                for code_block in data.get(
                    "code_blocks",
                    [],
                )
            ],
            links=[
                DocumentLink.from_dict(link)
                for link in data.get(
                    "links",
                    [],
                )
            ],
            page_number=data.get("page_number"),
            position=int(data.get("position", 0)),
            heading_path=[
                str(heading)
                for heading in data.get(
                    "heading_path",
                    [],
                )
            ],
            metadata=dict(
                data.get("metadata", {})
            ),
        )


@dataclass
class ParsedDocument:
    """
    Modèle commun produit par tous les parsers.

    Les futurs parsers :
    - HtmlParser ;
    - MarkdownParser ;
    - PdfParser ;

    retourneront tous un ParsedDocument.
    """

    document_id: str
    source_id: str
    source_type: str
    local_path: str
    checksum_sha256: str

    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    source_url: str | None = None

    language: str | None = None
    organization: str | None = None
    document_format: str | None = None

    sections: list[DocumentSection] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError(
                "document_id ne peut pas être vide."
            )

        if not self.source_id.strip():
            raise ValueError(
                "source_id ne peut pas être vide."
            )

        if not self.local_path.strip():
            raise ValueError(
                "local_path ne peut pas être vide."
            )

        if not self.checksum_sha256.strip():
            raise ValueError(
                "checksum_sha256 ne peut pas être vide."
            )

    @property
    def path(self) -> Path:
        """
        Retourne local_path sous forme d'objet Path.
        """

        return Path(self.local_path)

    @property
    def text(self) -> str:
        """
        Reconstruit le texte complet du document.
        """

        return "\n\n".join(
            section.text
            for section in sorted(
                self.sections,
                key=lambda section: section.position,
            )
            if not section.is_empty
        )

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def is_empty(self) -> bool:
        return not bool(self.text.strip())

    def add_section(
        self,
        section: DocumentSection,
    ) -> None:
        """
        Ajoute une section au document.
        """

        existing_ids = {
            existing_section.section_id
            for existing_section in self.sections
        }

        if section.section_id in existing_ids:
            raise ValueError(
                "Une section avec cet identifiant existe déjà : "
                f"{section.section_id}"
            )

        self.sections.append(section)

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit le document en dictionnaire JSON-compatible.
        """

        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "local_path": self.local_path,
            "checksum_sha256": self.checksum_sha256,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_url": self.source_url,
            "language": self.language,
            "organization": self.organization,
            "document_format": self.document_format,
            "sections": [
                section.to_dict()
                for section in self.sections
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ParsedDocument":
        """
        Reconstruit un document depuis un dictionnaire.
        """

        return cls(
            document_id=str(data["document_id"]),
            source_id=str(data["source_id"]),
            source_type=str(data["source_type"]),
            local_path=str(data["local_path"]),
            checksum_sha256=str(
                data["checksum_sha256"]
            ),
            title=data.get("title"),
            author=data.get("author"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            source_url=data.get("source_url"),
            language=data.get("language"),
            organization=data.get("organization"),
            document_format=data.get(
                "document_format"
            ),
            sections=[
                DocumentSection.from_dict(section)
                for section in data.get(
                    "sections",
                    [],
                )
            ],
            metadata=dict(
                data.get("metadata", {})
            ),
        )