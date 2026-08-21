from __future__ import annotations

import traceback

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from src.api.dependencies import (
    get_conversational_service,
)
from src.conversation import ConversationalRAGService
from src.api.schemas import (
    AskRequest,
    AskResponse,
    SourceResponse,
    TimingResponse,
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
    service: ConversationalRAGService = Depends(
        get_conversational_service
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
        "auto",
        "en",
        "fr",
        "ar",
    }:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "La langue doit être 'auto', 'en', 'fr' ou 'ar'."
            ),
        )

    reply_language = (
        str(payload.reply_language).strip().lower()
        if payload.reply_language is not None
        else None
    )
    if reply_language is not None and reply_language not in {
        "auto", "en", "fr", "ar"
    }:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="reply_language doit être 'auto', 'en', 'fr' ou 'ar'.",
        )

    response_style = str(payload.response_style or "concise").strip().lower()
    if response_style not in {"concise", "detailed", "expert"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="response_style doit être 'concise', 'detailed' ou 'expert'.",
        )

    try:
        # ====================================================
        # Configuration dynamique de la langue
        # ====================================================

        turn = service.process(
            question=question,
            conversation_id=payload.conversation_id,
            language=language,
            reply_language=reply_language,
            response_style=response_style,
        )
        response = turn.rag_response

        if response is None:
            return AskResponse(
                accepted=True,
                answer=turn.answer,
                provider=None,
                model_name=None,
                confidence_score=1.0,
                failed_rules=[],
                citations=[],
                sources=[],
                timings=TimingResponse(
                    retrieval_time_ms=0.0,
                    confidence_time_ms=0.0,
                    generation_time_ms=0.0,
                    citation_validation_time_ms=0.0,
                    total_time_ms=0.0,
                ),
                refusal_reason=None,
                original_query=turn.original_query,
                normalized_query=turn.normalized_query,
                rewritten_query=turn.rewritten_query,
                detected_language=turn.detected_language,
                reply_language=turn.reply_language,
                intent=turn.intent.value,
                conversation_id=turn.conversation_id,
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
            original_query=turn.original_query,
            normalized_query=turn.normalized_query,
            rewritten_query=turn.rewritten_query,
            detected_language=turn.detected_language,
            reply_language=turn.reply_language,
            intent=turn.intent.value,
            conversation_id=turn.conversation_id,
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
