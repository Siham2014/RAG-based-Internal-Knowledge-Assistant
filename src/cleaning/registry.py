from __future__ import annotations

from typing import Any, Type

from src.cleaning.base import (
    BaseCleaner,
    CleaningError,
)


class CleanerRegistry:
    """
    Registre central des cleaners disponibles.

    Exemple :

    @CleanerRegistry.register("whitespace")
    class WhitespaceCleaner(BaseCleaner):
        ...
    """

    _cleaners: dict[
        str,
        Type[BaseCleaner],
    ] = {}

    @classmethod
    def register(
        cls,
        *names: str,
    ):
        """
        Décorateur permettant d'enregistrer un cleaner
        sous un ou plusieurs noms.
        """

        if not names:
            raise CleaningError(
                "Au moins un nom doit être fourni "
                "pour enregistrer un cleaner."
            )

        normalized_names = [
            cls.normalize_name(name)
            for name in names
        ]

        def decorator(
            cleaner_class: Type[BaseCleaner],
        ) -> Type[BaseCleaner]:
            if not issubclass(
                cleaner_class,
                BaseCleaner,
            ):
                raise CleaningError(
                    "La classe enregistrée doit hériter "
                    "de BaseCleaner."
                )

            for normalized_name in normalized_names:
                existing_class = cls._cleaners.get(
                    normalized_name
                )

                if (
                    existing_class is not None
                    and existing_class is not cleaner_class
                ):
                    raise CleaningError(
                        "Un cleaner est déjà enregistré "
                        f"sous le nom '{normalized_name}' : "
                        f"{existing_class.__name__}"
                    )

                cls._cleaners[
                    normalized_name
                ] = cleaner_class

            return cleaner_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> BaseCleaner:
        """
        Crée une instance de cleaner à partir de son nom.
        """

        cleaner_class = cls.get_cleaner_class(
            name
        )

        return cleaner_class(
            enabled=enabled,
            config=config,
        )

    @classmethod
    def get_cleaner_class(
        cls,
        name: str,
    ) -> Type[BaseCleaner]:
        """
        Retourne la classe associée à un nom.
        """

        normalized_name = cls.normalize_name(
            name
        )

        cleaner_class = cls._cleaners.get(
            normalized_name
        )

        if cleaner_class is None:
            available = ", ".join(
                cls.available_names()
            )

            raise CleaningError(
                f"Aucun cleaner enregistré sous "
                f"le nom '{normalized_name}'. "
                f"Cleaners disponibles : "
                f"{available or 'aucun'}."
            )

        return cleaner_class

    @classmethod
    def supports(
        cls,
        name: str,
    ) -> bool:
        """
        Indique si un cleaner existe dans le registre.
        """

        try:
            normalized_name = cls.normalize_name(
                name
            )

        except CleaningError:
            return False

        return normalized_name in cls._cleaners

    @classmethod
    def available_names(
        cls,
    ) -> list[str]:
        """
        Retourne la liste triée des noms disponibles.
        """

        return sorted(
            cls._cleaners.keys()
        )

    @classmethod
    def available_cleaners(
        cls,
    ) -> dict[str, str]:
        """
        Retourne un mapping :
        nom du cleaner -> nom de sa classe.
        """

        return {
            name: cleaner_class.__name__
            for name, cleaner_class
            in sorted(
                cls._cleaners.items()
            )
        }

    @classmethod
    def clear(
        cls,
    ) -> None:
        """
        Vide le registre.

        Cette méthode est surtout utile dans les tests.
        """

        cls._cleaners.clear()

    @staticmethod
    def normalize_name(
        name: str,
    ) -> str:
        """
        Normalise le nom d'un cleaner.

        Exemples :

        " Whitespace " -> "whitespace"
        "unicode-cleaner" -> "unicode_cleaner"
        """

        if not isinstance(name, str):
            raise CleaningError(
                "Le nom du cleaner doit être "
                "une chaîne de caractères."
            )

        normalized_name = (
            name.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in normalized_name:
            normalized_name = (
                normalized_name.replace(
                    "__",
                    "_",
                )
            )

        normalized_name = normalized_name.strip(
            "_"
        )

        if not normalized_name:
            raise CleaningError(
                "Le nom du cleaner ne peut pas "
                "être vide."
            )

        return normalized_name