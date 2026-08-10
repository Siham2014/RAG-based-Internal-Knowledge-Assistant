from __future__ import annotations

from functools import lru_cache

from src.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """
    Construit le pipeline une seule fois.

    Les modèles d'embedding et de reranking ne seront donc
    pas rechargés pour chaque requête HTTP.
    """

    return RAGPipeline.from_settings(
        language="en"
    )


def shutdown_rag_pipeline() -> None:
    """
    Libère les modèles lors de l'arrêt de FastAPI.
    """

    pipeline = get_rag_pipeline()

    pipeline.unload_models()

    get_rag_pipeline.cache_clear()