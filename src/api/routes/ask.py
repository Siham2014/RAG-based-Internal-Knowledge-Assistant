from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.api.dependencies import (
    get_rag_pipeline,
)
from src.api.schemas import (
    AskRequest,
    AskResponse,
    SourceResponse,
    TimingResponse,
)
from src.pipeline import (
    RAGPipeline,
)


router = APIRouter(
    prefix="/ask",
    tags=["rag"],
)


@router.post(
    "",
    response_model=AskResponse,
)
def ask(
    payload: AskRequest,
    pipeline: RAGPipeline = Depends(
        get_rag_pipeline
    ),
) -> AskResponse:
    """
    Exécute le pipeline RAG complet.

    Attention :
    si le Confidence Gate accepte la question,
    cette route peut appeler le fournisseur LLM
    configuré et donc consommer des crédits.
    """

    question = str(
        payload.question
    ).strip()

    if not question:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "La question ne peut pas être vide."
            ),
        )

    try:
        # La langue est configurable par requête.
        pipeline.language = (
            str(
                payload.language
                or "en"
            )
            .strip()
            .lower()
        )

        response = pipeline.answer(
            question
        )

    except Exception as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Erreur pendant l'exécution "
                "du pipeline RAG : "
                f"{type(error).__name__}: {error}"
            ),
        ) from error

    sources = [
        SourceResponse(
            rank=source.rank,
            source=source.source,
            chunk_id=source.chunk_id,
            page_number=source.page_number,
            source_url=source.source_url,
            reranker_score=(
                source.reranker_score
            ),
            rrf_score=(
                source.rrf_score
            ),
        )
        for source in response.sources
    ]

    timings = TimingResponse(
        retrieval_time_ms=(
            response
            .timings
            .retrieval_time_ms
        ),
        confidence_time_ms=(
            response
            .timings
            .confidence_time_ms
        ),
        generation_time_ms=(
            response
            .timings
            .generation_time_ms
        ),
        citation_validation_time_ms=(
            response
            .timings
            .citation_validation_time_ms
        ),
        total_time_ms=(
            response
            .timings
            .total_time_ms
        ),
    )

    return AskResponse(
        accepted=response.accepted,
        answer=response.answer,
        provider=response.provider,
        model_name=response.model_name,
        confidence_score=(
            response
            .confidence
            .confidence_score
        ),
        failed_rules=list(
            response
            .confidence
            .failed_rules
        ),
        citations=list(
            response.citations
        ),
        sources=sources,
        timings=timings,
        refusal_reason=(
            response.refusal_reason
        ),
    )