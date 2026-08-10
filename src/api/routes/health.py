from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from src.api.dependencies import (
    get_rag_pipeline,
)
from src.api.schemas import (
    HealthResponse,
)
from src.pipeline import (
    RAGPipeline,
)


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
def health(
    pipeline: RAGPipeline = Depends(
        get_rag_pipeline
    ),
) -> HealthResponse:
    """
    Vérifie que l'API et le pipeline RAG
    sont correctement initialisés.

    Aucun appel LLM payant n'est effectué.
    """

    provider = None
    model_name = None
    rag_ready = False

    try:
        provider = (
            pipeline
            .llm_manager
            .provider_name
        )

        model_name = (
            pipeline
            .llm_manager
            .model_name
        )

        health_status = (
            pipeline
            .llm_manager
            .health_check()
        )

        rag_ready = bool(
            health_status
        )

    except Exception:
        rag_ready = False

    return HealthResponse(
        status=(
            "ok"
            if rag_ready
            else "degraded"
        ),
        rag_ready=rag_ready,
        provider=provider,
        model_name=model_name,
    )