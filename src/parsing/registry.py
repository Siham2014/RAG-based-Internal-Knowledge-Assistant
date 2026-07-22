from __future__ import annotations

from pathlib import Path
from typing import Type

from src.models.source import SourceConfig
from src.parsing.base import BaseParser, ParserError


class ParserRegistry:
    """
    Registre central des parsers disponibles.

    Il associe une extension de fichier à une classe de parser.
    """

    _parsers: dict[str, Type[BaseParser]] = {}

    @classmethod
    def register(
        cls,
        *extensions: str,
    ):
        """
        Décorateur utilisé pour enregistrer un parser.

        Exemple :

        @ParserRegistry.register(".md", ".markdown")
        class MarkdownParser(BaseParser):
            ...
        """

        def decorator(
            parser_class: Type[BaseParser],
        ) -> Type[BaseParser]:

            if not issubclass(
                parser_class,
                BaseParser,
            ):
                raise TypeError(
                    "Le parser enregistré doit hériter "
                    "de BaseParser."
                )

            if not extensions:
                raise ValueError(
                    "Au moins une extension doit être fournie."
                )

            normalized_extensions: set[str] = set()

            for extension in extensions:
                normalized = cls.normalize_extension(
                    extension
                )

                if normalized in cls._parsers:
                    existing_parser = (
                        cls._parsers[normalized]
                    )

                    raise ValueError(
                        f"L'extension '{normalized}' est déjà "
                        f"associée à "
                        f"{existing_parser.__name__}."
                    )

                normalized_extensions.add(normalized)

            for extension in normalized_extensions:
                cls._parsers[extension] = parser_class

            parser_class.supported_extensions = (
                normalized_extensions
            )

            return parser_class

        return decorator

    @classmethod
    def create(
        cls,
        file_path: Path,
        source: SourceConfig,
    ) -> BaseParser:
        """
        Crée le parser correspondant à l'extension du fichier.
        """

        parser_class = cls.get_parser_class(
            file_path
        )

        return parser_class(source=source)

    @classmethod
    def get_parser_class(
        cls,
        file_path: Path,
    ) -> Type[BaseParser]:
        """
        Retourne la classe du parser adaptée au fichier.
        """

        extension = cls.normalize_extension(
            file_path.suffix
        )

        parser_class = cls._parsers.get(extension)

        if parser_class is None:
            available = ", ".join(
                cls.available_extensions()
            )

            raise ParserError(
                f"Aucun parser enregistré pour "
                f"l'extension '{extension}'. "
                f"Extensions disponibles : "
                f"{available or 'aucune'}."
            )

        return parser_class

    @classmethod
    def supports(
        cls,
        file_path: Path,
    ) -> bool:
        """
        Indique si un parser existe pour le fichier.
        """

        extension = cls.normalize_extension(
            file_path.suffix
        )

        return extension in cls._parsers

    @classmethod
    def available_extensions(
        cls,
    ) -> list[str]:
        """
        Retourne les extensions enregistrées.
        """

        return sorted(cls._parsers.keys())

    @classmethod
    def available_parsers(
        cls,
    ) -> dict[str, str]:
        """
        Retourne les associations extension/parser.
        """

        return {
            extension: parser_class.__name__
            for extension, parser_class in sorted(
                cls._parsers.items()
            )
        }

    @classmethod
    def clear(cls) -> None:
        """
        Vide le registre.

        Cette méthode est principalement utile dans les tests.
        """

        cls._parsers.clear()

    @staticmethod
    def normalize_extension(
        extension: str,
    ) -> str:
        """
        Normalise une extension.

        Exemples :
        md   → .md
        MD   → .md
        .PDF → .pdf
        """

        value = str(extension).strip().lower()

        if not value:
            raise ValueError(
                "L'extension ne peut pas être vide."
            )

        if not value.startswith("."):
            value = f".{value}"

        return value