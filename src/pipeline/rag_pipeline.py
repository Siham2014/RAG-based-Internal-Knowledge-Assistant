from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.common.settings import (
    ApplicationSettings,
    get_settings,
)
from src.confidence import (
    ConfidenceDecision,
    ConfidenceGate,
    ConfidenceThresholds,
)
from src.generation import (
    CitationValidationResult,
    CitationValidator,
    GenerationContext,
    GenerationRequest,
    GenerationResponse,
    LLMGenerationError,
    LLMManager,
    LLMProviderFactory,
    REFUSAL_MESSAGE_EN,
    REFUSAL_MESSAGE_FR,
)
from src.pipeline.retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalPipelineResponse,
)
from src.reranking import (
    RerankedSearchResult,
)


# ============================================================
# Valeurs par défaut
# ============================================================

DEFAULT_LANGUAGE = "fr"
DEFAULT_GENERATION_CONTEXTS = 3
DEFAULT_MAX_ATTEMPTS_PER_PROVIDER = 2
DEFAULT_RETRY_DELAY_SECONDS = 1.0


# ============================================================
# Modèles de réponse
# ============================================================

@dataclass(frozen=True)
class RAGTimings:
    """
    Temps d'exécution des principales étapes du pipeline RAG.
    """

    retrieval_time_ms: float
    confidence_time_ms: float
    generation_time_ms: float
    citation_validation_time_ms: float
    total_time_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_time_ms": self.retrieval_time_ms,
            "confidence_time_ms": self.confidence_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "citation_validation_time_ms": (
                self.citation_validation_time_ms
            ),
            "total_time_ms": self.total_time_ms,
        }


@dataclass(frozen=True)
class RAGSource:
    """
    Source documentaire transmise à la couche de génération.
    """

    rank: int
    source: str
    chunk_id: str
    citation_id: str

    content: str
    source_url: str | None

    document_format: str | None
    page_number: int | None

    reranker_score: float
    rrf_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "content": self.content,
            "source_url": self.source_url,
            "document_format": self.document_format,
            "page_number": self.page_number,
            "reranker_score": self.reranker_score,
            "rrf_score": self.rrf_score,
        }


@dataclass(frozen=True)
class RAGResponse:
    """
    Réponse finale uniforme du système RAG.

    Une réponse peut être :

    - acceptée et générée ;
    - refusée par le Confidence Gate ;
    - refusée après l'échec du fournisseur LLM ;
    - refusée après l'échec de validation des citations.
    """

    question: str
    accepted: bool
    answer: str

    confidence: ConfidenceDecision

    sources: tuple[RAGSource, ...]
    citations: tuple[str, ...]

    provider: str | None
    model_name: str | None

    citation_validation: CitationValidationResult | None

    retrieval: RetrievalPipelineResponse
    generation: GenerationResponse | None

    timings: RAGTimings

    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "accepted": self.accepted,
            "answer": self.answer,
            "refusal_reason": self.refusal_reason,
            "confidence": self.confidence.to_dict(),
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "citations": list(self.citations),
            "provider": self.provider,
            "model_name": self.model_name,
            "citation_validation": (
                self.citation_validation.to_dict()
                if self.citation_validation is not None
                else None
            ),
            "retrieval": self.retrieval.to_dict(),
            "generation": (
                self.generation.to_dict()
                if self.generation is not None
                else None
            ),
            "timings": self.timings.to_dict(),
        }


# ============================================================
# Pipeline RAG principal
# ============================================================

class RAGPipeline:
    """
    Pipeline complet du système RAG.

    Étapes :

    1. Recherche documentaire hybride.
    2. Reranking Cross-Encoder.
    3. Évaluation du Confidence Gate.
    4. Refus si les preuves sont insuffisantes.
    5. Génération via LLMManager.
    6. Validation stricte des citations.
    7. Retour d'une réponse normalisée.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        confidence_gate: ConfidenceGate,
        llm_manager: LLMManager,
        citation_validator: CitationValidator,
        generation_enabled: bool = True,
        generation_contexts: int = DEFAULT_GENERATION_CONTEXTS,
        language: str = DEFAULT_LANGUAGE,
        max_output_tokens: int = 250,
        temperature: float = 0.0,
    ) -> None:
        if not isinstance(
            retrieval_pipeline,
            RetrievalPipeline,
        ):
            raise TypeError(
                "retrieval_pipeline doit être une instance "
                "de RetrievalPipeline."
            )

        if not isinstance(
            confidence_gate,
            ConfidenceGate,
        ):
            raise TypeError(
                "confidence_gate doit être une instance "
                "de ConfidenceGate."
            )

        if not isinstance(
            llm_manager,
            LLMManager,
        ):
            raise TypeError(
                "llm_manager doit être une instance "
                "de LLMManager."
            )

        if not isinstance(
            citation_validator,
            CitationValidator,
        ):
            raise TypeError(
                "citation_validator doit être une instance "
                "de CitationValidator."
            )

        if not isinstance(
            generation_enabled,
            bool,
        ):
            raise TypeError(
                "generation_enabled doit être un booléen."
            )

        try:
            normalized_generation_contexts = int(
                generation_contexts
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "generation_contexts doit être un entier."
            ) from error

        if normalized_generation_contexts <= 0:
            raise ValueError(
                "generation_contexts doit être supérieur à zéro."
            )

        try:
            normalized_max_output_tokens = int(
                max_output_tokens
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "max_output_tokens doit être un entier."
            ) from error

        if normalized_max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens doit être supérieur à zéro."
            )

        try:
            normalized_temperature = float(
                temperature
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "temperature doit être un nombre."
            ) from error

        if not 0.0 <= normalized_temperature <= 2.0:
            raise ValueError(
                "temperature doit être comprise entre 0 et 2."
            )

        normalized_language = (
            str(language)
            .strip()
            .lower()
        )

        if normalized_language not in {
            "fr",
            "en",
        }:
            raise ValueError(
                "language doit être 'fr' ou 'en'."
            )

        self.retrieval_pipeline = retrieval_pipeline
        self.confidence_gate = confidence_gate
        self.llm_manager = llm_manager
        self.citation_validator = citation_validator

        self.generation_enabled = generation_enabled
        self.generation_contexts = (
            normalized_generation_contexts
        )
        self.language = normalized_language
        self.max_output_tokens = (
            normalized_max_output_tokens
        )
        self.temperature = normalized_temperature

    # ========================================================
    # Construction depuis settings.yaml
    # ========================================================

    @classmethod
    def from_settings(
        cls,
        settings: ApplicationSettings | None = None,
        language: str = DEFAULT_LANGUAGE,
    ) -> "RAGPipeline":
        """
        Construit tous les composants depuis la configuration.
        """

        application_settings = (
            settings
            if settings is not None
            else get_settings()
        )

        confidence_settings = (
            application_settings.confidence
        )

        if not confidence_settings.enabled:
            raise ValueError(
                "confidence.enabled doit être true "
                "pour construire le RAGPipeline."
            )

        if (
            confidence_settings.minimum_top1_score
            is None
        ):
            raise ValueError(
                "confidence.minimum_top1_score est requis."
            )

        if (
            confidence_settings.minimum_margin
            is None
        ):
            raise ValueError(
                "confidence.minimum_margin est requis."
            )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        retrieval_pipeline = (
            RetrievalPipeline.from_settings(
                application_settings
            )
        )

        # ----------------------------------------------------
        # Confidence Gate
        # ----------------------------------------------------

        thresholds = ConfidenceThresholds(
            minimum_top1_score=(
                confidence_settings
                .minimum_top1_score
            ),
            minimum_margin=(
                confidence_settings
                .minimum_margin
            ),
            minimum_cosine_similarity=(
                confidence_settings
                .minimum_cosine_similarity
            ),
            minimum_rrf_score=(
                confidence_settings
                .minimum_rrf_score
            ),
            minimum_supporting_results=(
                confidence_settings
                .minimum_supporting_results
            ),
        )

        confidence_gate = ConfidenceGate(
            thresholds=thresholds
        )

        # ----------------------------------------------------
        # Fournisseur de génération
        # ----------------------------------------------------

        generation_settings = (
            application_settings.generation
        )

        provider_kwargs: dict[str, Any] = {}

        if generation_settings.model_name:
            provider_kwargs["model_name"] = (
                generation_settings.model_name
            )

        normalized_provider_name = (
            generation_settings.provider
            .strip()
            .lower()
        )

        if (
            normalized_provider_name == "huggingface"
            and generation_settings.inference_provider
        ):
            provider_kwargs["provider"] = (
                generation_settings.inference_provider
            )

        primary_provider = (
            LLMProviderFactory.create(
                provider_name=(
                    generation_settings.provider
                ),
                **provider_kwargs,
            )
        )

        llm_manager = LLMManager(
            primary_provider=primary_provider,
            fallback_providers=(),
            max_attempts_per_provider=(
                DEFAULT_MAX_ATTEMPTS_PER_PROVIDER
            ),
            retry_delay_seconds=(
                DEFAULT_RETRY_DELAY_SECONDS
            ),
        )

        # ----------------------------------------------------
        # Validation des citations
        # ----------------------------------------------------

        citation_validator = CitationValidator(
            require_at_least_one_citation=True
        )

        generation_contexts = min(
            DEFAULT_GENERATION_CONTEXTS,
            application_settings
            .retrieval
            .final_top_k,
        )

        return cls(
            retrieval_pipeline=retrieval_pipeline,
            confidence_gate=confidence_gate,
            llm_manager=llm_manager,
            citation_validator=citation_validator,
            generation_enabled=(
                generation_settings.enabled
            ),
            generation_contexts=(
                generation_contexts
            ),
            language=language,
            max_output_tokens=(
                generation_settings.max_tokens
            ),
            temperature=(
                generation_settings.temperature
            ),
        )

    # ========================================================
    # Conversion des résultats
    # ========================================================

    @staticmethod
    def _extract_source_url(
        result: RerankedSearchResult,
    ) -> str | None:
        """
        Extrait une URL éventuelle depuis les métadonnées.
        """

        metadata = result.metadata

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        possible_keys = (
            "url",
            "source_url",
            "canonical_url",
            "document_url",
            "web_url",
        )

        for key in possible_keys:
            value = metadata.get(
                key
            )

            if value is None:
                continue

            normalized = str(
                value
            ).strip()

            if normalized:
                return normalized

        return None

    @classmethod
    def _to_generation_context(
        cls,
        result: RerankedSearchResult,
    ) -> GenerationContext:
        return GenerationContext(
            rank=result.rank,
            chunk_id=result.chunk_id,
            source=result.source,
            content=result.content,
            source_url=(
                cls._extract_source_url(
                    result
                )
            ),
            document_format=(
                result.document_format
            ),
            page_number=(
                result.page_number
            ),
            reranker_score=(
                result.reranker_score
            ),
        )

    @classmethod
    def _to_rag_source(
        cls,
        result: RerankedSearchResult,
    ) -> RAGSource:
        context = cls._to_generation_context(
            result
        )

        return RAGSource(
            rank=result.rank,
            source=result.source,
            chunk_id=result.chunk_id,
            citation_id=context.citation_id,
            content=result.content,
            source_url=context.source_url,
            document_format=(
                result.document_format
            ),
            page_number=(
                result.page_number
            ),
            reranker_score=float(
                result.reranker_score
            ),
            rrf_score=float(
                result.rrf_score
            ),
        )

    def _refusal_message(self) -> str:
        if self.language == "en":
            return REFUSAL_MESSAGE_EN

        return REFUSAL_MESSAGE_FR

    @staticmethod
    def _build_timings(
        retrieval_time_ms: float,
        confidence_time_ms: float,
        generation_time_ms: float,
        citation_validation_time_ms: float,
        total_time_ms: float,
    ) -> RAGTimings:
        return RAGTimings(
            retrieval_time_ms=round(
                retrieval_time_ms,
                4,
            ),
            confidence_time_ms=round(
                confidence_time_ms,
                4,
            ),
            generation_time_ms=round(
                generation_time_ms,
                4,
            ),
            citation_validation_time_ms=round(
                citation_validation_time_ms,
                4,
            ),
            total_time_ms=round(
                total_time_ms,
                4,
            ),
        )

    # ========================================================
    # Exécution principale
    # ========================================================

    def answer(
        self,
        question: str,
        document_format: str | None = None,
        final_top_k: int | None = None,
    ) -> RAGResponse:
        """
        Exécute l'ensemble du pipeline RAG.
        """

        normalized_question = str(
            question
        ).strip()

        if not normalized_question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # 1. Retrieval et reranking
        # ----------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieval_response = (
            self.retrieval_pipeline.retrieve(
                question=normalized_question,
                document_format=document_format,
                final_top_k=final_top_k,
            )
        )

        retrieval_time_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        # ----------------------------------------------------
        # 2. Confidence Gate
        # ----------------------------------------------------

        confidence_start = time.perf_counter()

        confidence_decision = (
            self.confidence_gate.evaluate(
                retrieval_response.results
            )
        )

        confidence_time_ms = (
            time.perf_counter()
            - confidence_start
        ) * 1000

        selected_results = (
            retrieval_response.results[
                :self.generation_contexts
            ]
        )

        sources = tuple(
            self._to_rag_source(
                result
            )
            for result in selected_results
        )

        # ----------------------------------------------------
        # 3. Refus par le Confidence Gate
        # ----------------------------------------------------

        if not confidence_decision.accepted:
            total_time_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            failed_rules_text = ", ".join(
                confidence_decision.failed_rules
            )

            return RAGResponse(
                question=normalized_question,
                accepted=False,
                answer=self._refusal_message(),
                confidence=confidence_decision,
                sources=sources,
                citations=(),
                provider=None,
                model_name=None,
                citation_validation=None,
                retrieval=retrieval_response,
                generation=None,
                timings=self._build_timings(
                    retrieval_time_ms=(
                        retrieval_time_ms
                    ),
                    confidence_time_ms=(
                        confidence_time_ms
                    ),
                    generation_time_ms=0.0,
                    citation_validation_time_ms=0.0,
                    total_time_ms=total_time_ms,
                ),
                refusal_reason=(
                    "Confidence Gate rejection"
                    + (
                        f": {failed_rules_text}"
                        if failed_rules_text
                        else ""
                    )
                ),
            )

        # ----------------------------------------------------
        # 4. Génération désactivée
        # ----------------------------------------------------

        if not self.generation_enabled:
            total_time_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            return RAGResponse(
                question=normalized_question,
                accepted=True,
                answer=(
                    "La question est suffisamment documentée, "
                    "mais la génération est désactivée."
                ),
                confidence=confidence_decision,
                sources=sources,
                citations=(),
                provider=None,
                model_name=None,
                citation_validation=None,
                retrieval=retrieval_response,
                generation=None,
                timings=self._build_timings(
                    retrieval_time_ms=(
                        retrieval_time_ms
                    ),
                    confidence_time_ms=(
                        confidence_time_ms
                    ),
                    generation_time_ms=0.0,
                    citation_validation_time_ms=0.0,
                    total_time_ms=total_time_ms,
                ),
                refusal_reason=None,
            )

        if not selected_results:
            raise RuntimeError(
                "Le Confidence Gate a accepté la question, "
                "mais aucun contexte n'est disponible."
            )

        # ----------------------------------------------------
        # 5. Requête de génération
        # ----------------------------------------------------

        generation_contexts = tuple(
            self._to_generation_context(
                result
            )
            for result in selected_results
        )

        generation_request = GenerationRequest(
            question=normalized_question,
            contexts=generation_contexts,
            language=self.language,
            max_output_tokens=(
                self.max_output_tokens
            ),
            temperature=self.temperature,
        )

        # ----------------------------------------------------
        # 6. Appel via LLMManager
        # ----------------------------------------------------

        generation_start = time.perf_counter()

        try:
            generation_response = (
                self.llm_manager.generate(
                    generation_request
                )
            )

        except LLMGenerationError as error:
            generation_time_ms = (
                time.perf_counter()
                - generation_start
            ) * 1000

            total_time_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            return RAGResponse(
                question=normalized_question,
                accepted=False,
                answer=self._refusal_message(),
                confidence=confidence_decision,
                sources=sources,
                citations=(),
                provider=(
                    self.llm_manager
                    .provider_name
                ),
                model_name=(
                    self.llm_manager
                    .model_name
                ),
                citation_validation=None,
                retrieval=retrieval_response,
                generation=None,
                timings=self._build_timings(
                    retrieval_time_ms=(
                        retrieval_time_ms
                    ),
                    confidence_time_ms=(
                        confidence_time_ms
                    ),
                    generation_time_ms=(
                        generation_time_ms
                    ),
                    citation_validation_time_ms=0.0,
                    total_time_ms=total_time_ms,
                ),
                refusal_reason=(
                    "LLM generation failed: "
                    + str(error)
                ),
            )

        generation_time_ms = (
            time.perf_counter()
            - generation_start
        ) * 1000

        # ----------------------------------------------------
        # 7. Validation des citations
        # ----------------------------------------------------

        citation_start = time.perf_counter()

        citation_validation = (
            self.citation_validator
            .validate_response(
                response=generation_response,
                contexts=generation_contexts,
            )
        )

        citation_validation_time_ms = (
            time.perf_counter()
            - citation_start
        ) * 1000

        total_time_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        # ----------------------------------------------------
        # 8. Citations invalides
        # ----------------------------------------------------

        if not citation_validation.valid:
            return RAGResponse(
                question=normalized_question,
                accepted=False,
                answer=self._refusal_message(),
                confidence=confidence_decision,
                sources=sources,
                citations=(),
                provider=(
                    generation_response.provider
                ),
                model_name=(
                    generation_response.model_name
                ),
                citation_validation=(
                    citation_validation
                ),
                retrieval=retrieval_response,
                generation=generation_response,
                timings=self._build_timings(
                    retrieval_time_ms=(
                        retrieval_time_ms
                    ),
                    confidence_time_ms=(
                        confidence_time_ms
                    ),
                    generation_time_ms=(
                        generation_time_ms
                    ),
                    citation_validation_time_ms=(
                        citation_validation_time_ms
                    ),
                    total_time_ms=total_time_ms,
                ),
                refusal_reason=(
                    "Citation validation failed: "
                    + citation_validation.reason
                ),
            )

        # ----------------------------------------------------
        # 9. Réponse acceptée
        # ----------------------------------------------------

        return RAGResponse(
            question=normalized_question,
            accepted=True,
            answer=generation_response.answer,
            confidence=confidence_decision,
            sources=sources,
            citations=(
                citation_validation.valid_citations
            ),
            provider=(
                generation_response.provider
            ),
            model_name=(
                generation_response.model_name
            ),
            citation_validation=(
                citation_validation
            ),
            retrieval=retrieval_response,
            generation=generation_response,
            timings=self._build_timings(
                retrieval_time_ms=(
                    retrieval_time_ms
                ),
                confidence_time_ms=(
                    confidence_time_ms
                ),
                generation_time_ms=(
                    generation_time_ms
                ),
                citation_validation_time_ms=(
                    citation_validation_time_ms
                ),
                total_time_ms=total_time_ms,
            ),
            refusal_reason=None,
        )

    # ========================================================
    # Maintenance
    # ========================================================

    def unload_models(self) -> None:
        """
        Libère les modèles du pipeline de recherche.
        """

        self.retrieval_pipeline.unload_models()