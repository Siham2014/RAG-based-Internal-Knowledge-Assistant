from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.settings import PROJECT_ROOT
from src.pipeline import RAGPipeline, RAGResponse


# ============================================================
# Configuration du benchmark
# ============================================================

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "benchmarking"
    / "09_rag_question_suite"
)

DETAILS_PATH = (
    OUTPUT_DIRECTORY
    / "rag_question_suite_details.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "rag_question_suite_summary.json"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "rag_question_suite_manifest.json"
)


# ============================================================
# Jeu de questions
# ============================================================

@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    category: str
    question: str
    expected_answerable: bool
    language: str = "en"


QUESTIONS: tuple[EvaluationQuestion, ...] = (
    EvaluationQuestion(
        question_id="RAG001",
        category="azure_simple",
        question=(
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        ),
        expected_answerable=True,
    ),
    EvaluationQuestion(
        question_id="RAG002",
        category="azure_simple",
        question=(
            "What is a sustainable workload on Azure?"
        ),
        expected_answerable=True,
    ),
    EvaluationQuestion(
        question_id="RAG003",
        category="azure_simple",
        question=(
            "How can unnecessary resource consumption "
            "be reduced in an Azure workload?"
        ),
        expected_answerable=True,
    ),
    EvaluationQuestion(
        question_id="RAG004",
        category="azure_complex",
        question=(
            "How are sustainability, reliability, security, "
            "and performance related in Azure workloads?"
        ),
        expected_answerable=True,
    ),
    EvaluationQuestion(
        question_id="RAG005",
        category="azure_complex",
        question=(
            "Which Azure services can help organizations "
            "understand and reduce carbon emissions?"
        ),
        expected_answerable=True,
    ),
    EvaluationQuestion(
        question_id="RAG006",
        category="technical_out_of_corpus",
        question=(
            "How do I configure Cisco IOS "
            "BGP route reflection?"
        ),
        expected_answerable=False,
    ),
    EvaluationQuestion(
        question_id="RAG007",
        category="technical_out_of_corpus",
        question=(
            "How do I configure a VMware ESXi "
            "high-availability cluster?"
        ),
        expected_answerable=False,
    ),
    EvaluationQuestion(
        question_id="RAG008",
        category="technical_out_of_corpus",
        question=(
            "How do I create a Jenkins declarative pipeline?"
        ),
        expected_answerable=False,
    ),
    EvaluationQuestion(
        question_id="RAG009",
        category="general_out_of_domain",
        question=(
            "Who won the FIFA World Cup in 2022?"
        ),
        expected_answerable=False,
    ),
    EvaluationQuestion(
        question_id="RAG010",
        category="general_out_of_domain",
        question=(
            "How do I prepare homemade lasagna?"
        ),
        expected_answerable=False,
    ),
)


# ============================================================
# Modèle de résultat
# ============================================================

@dataclass
class QuestionResult:
    question_id: str
    category: str
    question: str
    expected_answerable: bool

    predicted_answerable: bool | None
    prediction_correct: bool

    accepted: bool | None
    answer: str

    confidence_score: float | None
    top1_reranker_score: float | None
    reranker_margin: float | None
    top1_cosine_similarity: float | None
    top1_rrf_score: float | None
    supporting_results_count: int | None
    failed_rules: str

    provider: str | None
    model_name: str | None
    llm_called: bool
    llm_attempts: int
    llm_success: bool

    citations_valid: bool | None
    citation_count: int
    citations: str

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None

    retrieval_time_ms: float | None
    confidence_time_ms: float | None
    generation_time_ms: float | None
    citation_validation_time_ms: float | None
    total_time_ms: float | None

    source_count: int
    top_source: str | None
    top_chunk_id: str | None

    refusal_reason: str | None
    error_type: str | None
    error_message: str | None


# ============================================================
# Fonctions utilitaires
# ============================================================

def safe_round(
    value: float | None,
    digits: int = 6,
) -> float | None:
    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.mean(values),
        4,
    )


def median_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.median(values),
        4,
    )


def build_success_result(
    item: EvaluationQuestion,
    response: RAGResponse,
    llm_called: bool,
    llm_attempts: int,
) -> QuestionResult:
    features = response.confidence.features

    generation = response.generation
    usage = (
        generation.usage
        if generation is not None
        else None
    )

    citation_validation = (
        response.citation_validation
    )

    top_source = (
        response.sources[0].source
        if response.sources
        else None
    )

    top_chunk_id = (
        response.sources[0].chunk_id
        if response.sources
        else None
    )

    predicted_answerable = bool(
        response.accepted
    )

    return QuestionResult(
        question_id=item.question_id,
        category=item.category,
        question=item.question,
        expected_answerable=(
            item.expected_answerable
        ),
        predicted_answerable=(
            predicted_answerable
        ),
        prediction_correct=(
            predicted_answerable
            == item.expected_answerable
        ),
        accepted=response.accepted,
        answer=response.answer,
        confidence_score=safe_round(
            response.confidence
            .confidence_score
        ),
        top1_reranker_score=safe_round(
            features.top1_reranker_score
        ),
        reranker_margin=safe_round(
            features.reranker_margin
        ),
        top1_cosine_similarity=safe_round(
            features.top1_cosine_similarity
        ),
        top1_rrf_score=safe_round(
            features.top1_rrf_score,
            8,
        ),
        supporting_results_count=(
            features.supporting_results_count
        ),
        failed_rules="|".join(
            response.confidence.failed_rules
        ),
        provider=response.provider,
        model_name=response.model_name,
        llm_called=llm_called,
        llm_attempts=llm_attempts,
        llm_success=(
            generation is not None
        ),
        citations_valid=(
            citation_validation.valid
            if citation_validation is not None
            else None
        ),
        citation_count=len(
            response.citations
        ),
        citations="|".join(
            response.citations
        ),
        input_tokens=(
            usage.input_tokens
            if usage is not None
            else None
        ),
        output_tokens=(
            usage.output_tokens
            if usage is not None
            else None
        ),
        total_tokens=(
            usage.total_tokens
            if usage is not None
            else None
        ),
        retrieval_time_ms=(
            response.timings
            .retrieval_time_ms
        ),
        confidence_time_ms=(
            response.timings
            .confidence_time_ms
        ),
        generation_time_ms=(
            response.timings
            .generation_time_ms
        ),
        citation_validation_time_ms=(
            response.timings
            .citation_validation_time_ms
        ),
        total_time_ms=(
            response.timings
            .total_time_ms
        ),
        source_count=len(
            response.sources
        ),
        top_source=top_source,
        top_chunk_id=top_chunk_id,
        refusal_reason=(
            response.refusal_reason
        ),
        error_type=None,
        error_message=None,
    )


def build_error_result(
    item: EvaluationQuestion,
    error: Exception,
    elapsed_time_ms: float,
) -> QuestionResult:
    return QuestionResult(
        question_id=item.question_id,
        category=item.category,
        question=item.question,
        expected_answerable=(
            item.expected_answerable
        ),
        predicted_answerable=None,
        prediction_correct=False,
        accepted=None,
        answer="",
        confidence_score=None,
        top1_reranker_score=None,
        reranker_margin=None,
        top1_cosine_similarity=None,
        top1_rrf_score=None,
        supporting_results_count=None,
        failed_rules="",
        provider=None,
        model_name=None,
        llm_called=False,
        llm_attempts=0,
        llm_success=False,
        citations_valid=None,
        citation_count=0,
        citations="",
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        retrieval_time_ms=None,
        confidence_time_ms=None,
        generation_time_ms=None,
        citation_validation_time_ms=None,
        total_time_ms=round(
            elapsed_time_ms,
            4,
        ),
        source_count=0,
        top_source=None,
        top_chunk_id=None,
        refusal_reason=None,
        error_type=type(error).__name__,
        error_message=str(error),
    )


# ============================================================
# Sauvegarde
# ============================================================

def save_details(
    results: list[QuestionResult],
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(result)
        for result in results
    ]

    if not rows:
        raise RuntimeError(
            "Aucun résultat à enregistrer."
        )

    with DETAILS_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def calculate_summary(
    results: list[QuestionResult],
) -> dict[str, Any]:
    valid_results = [
        result
        for result in results
        if result.accepted is not None
    ]

    expected_positive = [
        result
        for result in valid_results
        if result.expected_answerable
    ]

    expected_negative = [
        result
        for result in valid_results
        if not result.expected_answerable
    ]

    true_positive = sum(
        1
        for result in valid_results
        if (
            result.expected_answerable
            and result.predicted_answerable
        )
    )

    true_negative = sum(
        1
        for result in valid_results
        if (
            not result.expected_answerable
            and not result.predicted_answerable
        )
    )

    false_positive = sum(
        1
        for result in valid_results
        if (
            not result.expected_answerable
            and result.predicted_answerable
        )
    )

    false_negative = sum(
        1
        for result in valid_results
        if (
            result.expected_answerable
            and not result.predicted_answerable
        )
    )

    accuracy = (
        (true_positive + true_negative)
        / len(valid_results)
        if valid_results
        else 0.0
    )

    citation_checked = [
        result
        for result in results
        if result.citations_valid
        is not None
    ]

    citation_valid_count = sum(
        1
        for result in citation_checked
        if result.citations_valid
    )

    retrieval_times = [
        float(result.retrieval_time_ms)
        for result in results
        if result.retrieval_time_ms
        is not None
    ]

    generation_times = [
        float(result.generation_time_ms)
        for result in results
        if (
            result.generation_time_ms
            is not None
            and result.generation_time_ms > 0
        )
    ]

    total_times = [
        float(result.total_time_ms)
        for result in results
        if result.total_time_ms
        is not None
    ]

    total_input_tokens = sum(
        result.input_tokens or 0
        for result in results
    )

    total_output_tokens = sum(
        result.output_tokens or 0
        for result in results
    )

    total_tokens = sum(
        result.total_tokens or 0
        for result in results
    )

    return {
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "number_of_questions": len(
            results
        ),
        "number_of_completed_questions": len(
            valid_results
        ),
        "number_of_errors": sum(
            1
            for result in results
            if result.error_type is not None
        ),
        "expected_answerable_questions": len(
            expected_positive
        ),
        "expected_unanswerable_questions": len(
            expected_negative
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "classification_accuracy": round(
            accuracy,
            6,
        ),
        "accepted_questions": sum(
            1
            for result in valid_results
            if result.accepted
        ),
        "rejected_questions": sum(
            1
            for result in valid_results
            if not result.accepted
        ),
        "llm_calls": sum(
            1
            for result in results
            if result.llm_called
        ),
        "successful_llm_generations": sum(
            1
            for result in results
            if result.llm_success
        ),
        "citation_validations_executed": len(
            citation_checked
        ),
        "valid_citation_responses": (
            citation_valid_count
        ),
        "citation_validity_rate": round(
            citation_valid_count
            / len(citation_checked),
            6,
        )
        if citation_checked
        else None,
        "total_input_tokens": (
            total_input_tokens
        ),
        "total_output_tokens": (
            total_output_tokens
        ),
        "total_tokens": total_tokens,
        "mean_retrieval_time_ms": (
            mean_or_none(
                retrieval_times
            )
        ),
        "median_retrieval_time_ms": (
            median_or_none(
                retrieval_times
            )
        ),
        "mean_generation_time_ms": (
            mean_or_none(
                generation_times
            )
        ),
        "median_generation_time_ms": (
            median_or_none(
                generation_times
            )
        ),
        "mean_total_time_ms": (
            mean_or_none(
                total_times
            )
        ),
        "median_total_time_ms": (
            median_or_none(
                total_times
            )
        ),
    }


def save_json(
    path: Path,
    content: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            content,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Affichage
# ============================================================

def print_result(
    index: int,
    total: int,
    result: QuestionResult,
) -> None:
    expected_label = (
        "answerable"
        if result.expected_answerable
        else "unanswerable"
    )

    predicted_label = (
        "accepted"
        if result.accepted is True
        else (
            "rejected"
            if result.accepted is False
            else "error"
        )
    )

    status = (
        "OK"
        if result.prediction_correct
        else "FAIL"
    )

    print(
        f"[{index:02d}/{total:02d}] "
        f"{result.question_id} | "
        f"{result.category} | "
        f"expected={expected_label} | "
        f"result={predicted_label} | "
        f"confidence={result.confidence_score} | "
        f"LLM={result.llm_called} | "
        f"citations={result.citations_valid} | "
        f"{status}"
    )

    if result.error_message:
        print(
            "    Erreur :",
            result.error_message,
        )


# ============================================================
# Exécution
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("BENCHMARK FONCTIONNEL DU PIPELINE RAG")
    print("=" * 110)
    print(
        "Questions :",
        len(QUESTIONS),
    )

    rag = RAGPipeline.from_settings(
        language="en"
    )

    # Une seule tentative distante par question pour éviter
    # de multiplier les appels Hugging Face.
    rag.llm_manager.max_attempts_per_provider = 1
    rag.llm_manager.retry_delay_seconds = 0.0

    print(
        "Provider :",
        rag.llm_manager.provider_name,
    )
    print(
        "Modèle :",
        rag.llm_manager.model_name,
    )
    print(
        "Health :",
        rag.llm_manager.health_check(),
    )
    print()

    results: list[QuestionResult] = []

    try:
        for index, item in enumerate(
            QUESTIONS,
            start=1,
        ):
            before_attempts = (
                rag.llm_manager.last_attempts
            )

            start = time.perf_counter()

            try:
                response = rag.answer(
                    item.question
                )

                after_attempts = (
                    rag.llm_manager.last_attempts
                )

                llm_called = (
                    after_attempts
                    != before_attempts
                )

                current_attempts = (
                    after_attempts
                    if llm_called
                    else ()
                )

                result = build_success_result(
                    item=item,
                    response=response,
                    llm_called=llm_called,
                    llm_attempts=len(
                        current_attempts
                    ),
                )

            except Exception as error:
                elapsed_time_ms = (
                    time.perf_counter()
                    - start
                ) * 1000

                result = build_error_result(
                    item=item,
                    error=error,
                    elapsed_time_ms=(
                        elapsed_time_ms
                    ),
                )

            results.append(
                result
            )

            print_result(
                index=index,
                total=len(QUESTIONS),
                result=result,
            )

            # Sauvegarde après chaque question afin de ne pas
            # perdre les résultats en cas d'arrêt.
            save_details(
                results
            )

        summary = calculate_summary(
            results
        )

        manifest = {
            "benchmark_name": (
                "rag_question_suite"
            ),
            "created_at_utc": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "provider": (
                rag.llm_manager.provider_name
            ),
            "model_name": (
                rag.llm_manager.model_name
            ),
            "number_of_questions": len(
                QUESTIONS
            ),
            "questions": [
                asdict(question)
                for question in QUESTIONS
            ],
            "details_path": str(
                DETAILS_PATH
            ),
            "summary_path": str(
                SUMMARY_PATH
            ),
        }

        save_json(
            SUMMARY_PATH,
            summary,
        )

        save_json(
            MANIFEST_PATH,
            manifest,
        )

        print()
        print("=" * 110)
        print("RÉSUMÉ")
        print("=" * 110)

        for key, value in summary.items():
            print(
                f"{key} : {value}"
            )

        print()
        print(
            "Détails :",
            DETAILS_PATH,
        )
        print(
            "Résumé :",
            SUMMARY_PATH,
        )
        print(
            "Manifeste :",
            MANIFEST_PATH,
        )

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()