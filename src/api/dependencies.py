from __future__ import annotations

from functools import lru_cache

from src.pipeline import RAGPipeline
from src.common.settings import get_settings
from src.conversation import (
    ConversationQueryRewriter,
    ConversationalRAGService,
    InMemoryConversationMemory,
)


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


@lru_cache(maxsize=1)
def get_conversational_service() -> ConversationalRAGService:
    pipeline = get_rag_pipeline()
    settings = get_settings().conversation
    provider = (
        pipeline.query_normalizer.provider
        if pipeline.query_normalizer is not None
        else None
    )
    return ConversationalRAGService(
        rag_pipeline=pipeline,
        memory=InMemoryConversationMemory(
            max_conversations=settings.max_conversations,
            max_messages=settings.max_messages,
        ),
        query_rewriter=ConversationQueryRewriter(
            provider=provider if settings.enabled else None,
            max_output_tokens=settings.rewrite_max_tokens,
            max_context_messages=settings.max_context_messages,
            max_context_chars=settings.max_context_chars,
        ),
    )


def shutdown_rag_pipeline() -> None:
    """
    Libère les modèles lors de l'arrêt de FastAPI.
    """

    pipeline = get_rag_pipeline()

    pipeline.unload_models()

    get_rag_pipeline.cache_clear()
    get_conversational_service.cache_clear()
