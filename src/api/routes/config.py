from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import ConfigResponse
from src.common.settings import get_settings


router = APIRouter(
    prefix="/config",
    tags=["config"],
)


@router.get(
    "",
    response_model=ConfigResponse,
)
def config() -> ConfigResponse:
    """
    Retourne uniquement les paramètres non sensibles
    du pipeline RAG.

    Les clés API ne sont jamais exposées.
    """

    settings = get_settings()

    return ConfigResponse(
        project_name=settings.project.name,
        company=settings.corpus.company,
        embedding_model=settings.embedding.model_name,
        reranker_model=settings.reranking.model_name,
        candidate_k=settings.retrieval.candidate_k,
        hybrid_top_k=settings.retrieval.hybrid_top_k,
        final_top_k=settings.retrieval.final_top_k,
        confidence_enabled=settings.confidence.enabled,
        minimum_top1_score=(
            settings.confidence.minimum_top1_score
        ),
        generation_enabled=settings.generation.enabled,
        generation_provider=settings.generation.provider,
        generation_model=settings.generation.model_name,
        generation_context_count=(
            settings.generation.context_count
        ),
    )