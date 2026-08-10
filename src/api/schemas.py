from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question envoyée au pipeline RAG.",
    )

    language: str = Field(
        default="en",
        description="Langue de génération de la réponse.",
    )


class SourceResponse(BaseModel):
    rank: int
    source: str
    chunk_id: str
    page_number: int | None = None
    source_url: str | None = None
    reranker_score: float
    rrf_score: float


class TimingResponse(BaseModel):
    retrieval_time_ms: float
    confidence_time_ms: float
    generation_time_ms: float
    citation_validation_time_ms: float
    total_time_ms: float


class AskResponse(BaseModel):
    accepted: bool

    answer: str

    provider: str | None = None
    model_name: str | None = None

    confidence_score: float

    failed_rules: list[str]

    citations: list[str]

    sources: list[SourceResponse]

    timings: TimingResponse

    refusal_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    rag_ready: bool

    provider: str | None = None
    model_name: str | None = None


class ConfigResponse(BaseModel):
    project_name: str
    company: str

    embedding_model: str
    reranker_model: str

    candidate_k: int
    hybrid_top_k: int
    final_top_k: int

    confidence_enabled: bool
    minimum_top1_score: float | None

    generation_enabled: bool
    generation_provider: str
    generation_model: str | None
    generation_context_count: int