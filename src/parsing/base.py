from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path

from src.models.document import ParsedDocument
from src.models.source import SourceConfig


class ParserError(Exception):
    """Erreur levée lors du parsing."""


class BaseParser(ABC):
    """
    Classe abstraite utilisée par tous les parsers.
    """

    supported_extensions: set[str] = set()

    def __init__(self, source: SourceConfig):
        self.source = source

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse un fichier et retourne un ParsedDocument.
        """
        ...

    def validate_file(self, file_path: Path) -> None:
        """
        Vérifie que le fichier existe et possède
        une extension supportée.
        """

        if not file_path.exists():
            raise ParserError(
                f"Fichier introuvable : {file_path}"
            )

        if not file_path.is_file():
            raise ParserError(
                f"{file_path} n'est pas un fichier."
            )

        extension = file_path.suffix.lower()

        if (
            self.supported_extensions
            and extension not in self.supported_extensions
        ):
            raise ParserError(
                f"Extension non supportée : {extension}"
            )

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """
        Calcule le SHA-256 d'un fichier.
        """

        digest = sha256()

        with file_path.open("rb") as file:
            while chunk := file.read(65536):
                digest.update(chunk)

        return digest.hexdigest()

    def build_document_id(
        self,
        file_path: Path,
    ) -> str:
        """
        Génère un identifiant unique.
        """

        relative = file_path.stem.lower()

        return (
            f"{self.source.id}:"
            f"{relative}"
        )