from __future__ import annotations

import traceback

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

    Si le Confidence Gate accepte la question,
    cette route peut appeler le fournisseur LLM
    configuré et donc consommer des crédits.
    """

    question = str(
        payload.question or ""
    ).strip()

    language = str(
        payload.language or "en"
    ).strip().lower()

    if not question:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "La question ne peut pas être vide."
            ),
        )

    if language not in {
        "en",
        "fr",
    }:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "La langue doit être 'en' ou 'fr'."
            ),
        )

    try:
        # ====================================================
        # Configuration dynamique de la langue
        # ====================================================

        pipeline.language = language

        # ====================================================
        # Exécution complète du RAG
        # ====================================================

        response = pipeline.answer(
            question
        )

        # ====================================================
        # Conversion des sources
        # ====================================================

        sources = [
            SourceResponse(
                rank=source.rank,
                source=source.source,
                chunk_id=source.chunk_id,
                page_number=source.page_number,
                source_url=source.source_url,
                reranker_score=float(
                    source.reranker_score
                ),
                rrf_score=float(
                    source.rrf_score
                ),
            )
            for source in response.sources
        ]

        # ====================================================
        # Conversion des timings
        # ====================================================

        timings = TimingResponse(
            retrieval_time_ms=float(
                response
                .timings
                .retrieval_time_ms
            ),
            confidence_time_ms=float(
                response
                .timings
                .confidence_time_ms
            ),
            generation_time_ms=float(
                response
                .timings
                .generation_time_ms
            ),
            citation_validation_time_ms=float(
                response
                .timings
                .citation_validation_time_ms
            ),
            total_time_ms=float(
                response
                .timings
                .total_time_ms
            ),
        )

        # ====================================================
        # Réponse API finale
        # ====================================================

        return AskResponse(
            accepted=bool(
                response.accepted
            ),
            answer=str(
                response.answer or ""
            ),
            provider=(
                str(response.provider)
                if response.provider
                else None
            ),
            model_name=(
                str(response.model_name)
                if response.model_name
                else None
            ),
            confidence_score=float(
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
                str(
                    response.refusal_reason
                )
                if response.refusal_reason
                else None
            ),
        )

    except HTTPException:
        raise

    except Exception as error:
        print()
        print("=" * 100)
        print("ERREUR DANS POST /ask")
        print("=" * 100)
        print(
            "Question :",
            question,
        )
        print(
            "Language :",
            language,
        )
        print(
            "Type :",
            type(error).__name__,
        )
        print(
            "Message :",
            str(error),
        )
        print("-" * 100)

        traceback.print_exc()

        print("=" * 100)
        print()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "message": (
                    "Erreur pendant l'exécution "
                    "du pipeline RAG."
                ),
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            },
        ) from error