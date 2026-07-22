from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from src.models.document import ParsedDocument


class CleaningError(Exception):
    """
    Exception levée lorsqu'une erreur survient pendant
    le nettoyage d'un document.
    """


class BaseCleaner(ABC):
    """
    Classe abstraite de base pour tous les cleaners.

    Un cleaner reçoit un ParsedDocument et retourne
    une nouvelle instance nettoyée.

    Le document original ne doit jamais être modifié.
    """

    cleaner_name: str = "base_cleaner"

    def __init__(
        self,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.config = config or {}

    def __call__(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Permet d'utiliser directement une instance comme
        une fonction :

        cleaned_document = cleaner(document)
        """

        return self.run(document)

    def run(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Exécute le cleaner sur une copie du document.
        """

        self.validate_document(document)

        if not self.enabled:
            return deepcopy(document)

        document_copy = deepcopy(document)

        try:
            cleaned_document = self.clean(
                document_copy
            )

        except CleaningError:
            raise

        except Exception as error:
            raise CleaningError(
                f"Erreur dans le cleaner "
                f"'{self.cleaner_name}' : {error}"
            ) from error

        self.validate_result(cleaned_document)

        self._update_cleaning_metadata(
            cleaned_document
        )

        return cleaned_document

    @abstractmethod
    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """
        Méthode à implémenter dans chaque cleaner.

        Elle reçoit une copie du ParsedDocument et doit
        retourner le document nettoyé.
        """

        raise NotImplementedError

    @staticmethod
    def validate_document(
        document: ParsedDocument,
    ) -> None:
        """
        Vérifie que l'entrée est un ParsedDocument valide.
        """

        if not isinstance(
            document,
            ParsedDocument,
        ):
            raise CleaningError(
                "Le cleaner attend une instance de "
                "ParsedDocument."
            )

    @staticmethod
    def validate_result(
        document: ParsedDocument,
    ) -> None:
        """
        Vérifie que le cleaner retourne un ParsedDocument.
        """

        if not isinstance(
            document,
            ParsedDocument,
        ):
            raise CleaningError(
                "Le cleaner doit retourner une instance "
                "de ParsedDocument."
            )

    def _update_cleaning_metadata(
        self,
        document: ParsedDocument,
    ) -> None:
        """
        Ajoute le nom du cleaner exécuté dans les
        métadonnées du document.
        """

        if document.metadata is None:
            document.metadata = {}

        applied_cleaners = document.metadata.get(
            "applied_cleaners",
            [],
        )

        if not isinstance(
            applied_cleaners,
            list,
        ):
            applied_cleaners = []

        if self.cleaner_name not in applied_cleaners:
            applied_cleaners.append(
                self.cleaner_name
            )

        document.metadata[
            "applied_cleaners"
        ] = applied_cleaners

        document.metadata[
            "cleaning_applied"
        ] = True

    @staticmethod
    def clean_string(
        value: str | None,
    ) -> str:
        """
        Nettoyage minimal commun à plusieurs cleaners.

        Cette méthode :
        - accepte None ;
        - convertit la valeur en chaîne ;
        - retire les espaces au début et à la fin.
        """

        if value is None:
            return ""

        return str(value).strip()

    def get_config(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Récupère une option de configuration.
        """

        return self.config.get(
            key,
            default,
        )

    def describe(
        self,
    ) -> dict[str, Any]:
        """
        Retourne une description du cleaner.
        """

        return {
            "name": self.cleaner_name,
            "class": self.__class__.__name__,
            "enabled": self.enabled,
            "config": dict(self.config),
        }