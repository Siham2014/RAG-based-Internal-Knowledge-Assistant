from __future__ import annotations

from typing import TypeVar

from src.chunking.base import BaseChunker
from src.chunking.models import ChunkingConfig, ChunkingStrategy


ChunkerType = TypeVar("ChunkerType", bound=BaseChunker)


class ChunkerRegistryError(ValueError):
    """
    Erreur déclenchée lors de l'enregistrement ou de la création d'un chunker.
    """


class ChunkerRegistry:
    """
    Registre centralisé des stratégies de chunking.

    Il permet :
    - d'enregistrer une classe de chunker ;
    - de récupérer une classe ;
    - de créer une instance ;
    - de lister les stratégies disponibles.
    """

    _registry: dict[ChunkingStrategy, type[BaseChunker]] = {}

    @classmethod
    def register(
        cls,
        strategy: ChunkingStrategy | str,
        chunker_class: type[ChunkerType],
        *,
        replace: bool = False,
    ) -> None:
        normalized_strategy = cls._normalize_strategy(strategy)

        if not isinstance(chunker_class, type):
            raise ChunkerRegistryError(
                "chunker_class doit être une classe."
            )

        if not issubclass(chunker_class, BaseChunker):
            raise ChunkerRegistryError(
                "La classe enregistrée doit hériter de BaseChunker."
            )

        if normalized_strategy in cls._registry and not replace:
            existing = cls._registry[normalized_strategy]

            raise ChunkerRegistryError(
                f"La stratégie {normalized_strategy.value!r} est déjà "
                f"enregistrée avec {existing.__name__}."
            )

        declared_strategy = getattr(
            chunker_class,
            "strategy_name",
            None,
        )

        if declared_strategy is not None:
            if declared_strategy != normalized_strategy.value:
                raise ChunkerRegistryError(
                    f"La classe {chunker_class.__name__} déclare "
                    f"strategy_name={declared_strategy!r}, mais elle est "
                    f"enregistrée pour {normalized_strategy.value!r}."
                )

        cls._registry[normalized_strategy] = chunker_class

    @classmethod
    def unregister(
        cls,
        strategy: ChunkingStrategy | str,
    ) -> None:
        normalized_strategy = cls._normalize_strategy(strategy)

        cls._registry.pop(normalized_strategy, None)

    @classmethod
    def get(
        cls,
        strategy: ChunkingStrategy | str,
    ) -> type[BaseChunker]:
        normalized_strategy = cls._normalize_strategy(strategy)

        try:
            return cls._registry[normalized_strategy]
        except KeyError as exc:
            available = ", ".join(cls.available_strategies())

            if not available:
                available = "aucune"

            raise ChunkerRegistryError(
                f"Aucun chunker enregistré pour "
                f"{normalized_strategy.value!r}. "
                f"Stratégies disponibles : {available}."
            ) from exc

    @classmethod
    def create(
        cls,
        config: ChunkingConfig,
    ) -> BaseChunker:
        if not isinstance(config, ChunkingConfig):
            raise ChunkerRegistryError(
                "config doit être une instance de ChunkingConfig."
            )

        chunker_class = cls.get(config.strategy)

        return chunker_class(config=config)

    @classmethod
    def is_registered(
        cls,
        strategy: ChunkingStrategy | str,
    ) -> bool:
        normalized_strategy = cls._normalize_strategy(strategy)

        return normalized_strategy in cls._registry

    @classmethod
    def available_strategies(cls) -> list[str]:
        return sorted(
            strategy.value
            for strategy in cls._registry
        )

    @classmethod
    def clear(cls) -> None:
        """
        Supprime toutes les stratégies enregistrées.

        Cette méthode est surtout utile pour les tests.
        """

        cls._registry.clear()

    @staticmethod
    def _normalize_strategy(
        strategy: ChunkingStrategy | str,
    ) -> ChunkingStrategy:
        if isinstance(strategy, ChunkingStrategy):
            return strategy

        if isinstance(strategy, str):
            try:
                return ChunkingStrategy(strategy)
            except ValueError as exc:
                supported = ", ".join(
                    item.value for item in ChunkingStrategy
                )

                raise ChunkerRegistryError(
                    f"Stratégie inconnue : {strategy!r}. "
                    f"Valeurs autorisées : {supported}."
                ) from exc

        raise ChunkerRegistryError(
            "strategy doit être une chaîne ou une instance "
            "de ChunkingStrategy."
        )