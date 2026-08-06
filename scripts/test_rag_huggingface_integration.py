from __future__ import annotations

from src.pipeline import (
    RAGPipeline,
    RAGResponse,
)


ANSWERABLE_QUESTION = (
    "Who shares responsibility for the "
    "sustainability of cloud workloads?"
)

OUT_OF_CORPUS_QUESTION = (
    "How do I configure Cisco IOS "
    "BGP route reflection?"
)


def print_response(
    response: RAGResponse,
) -> None:
    print()
    print("Accepted :", response.accepted)
    print("Provider :", response.provider)
    print("Model :", response.model_name)

    print()
    print("Answer")
    print("-" * 100)
    print(response.answer)

    print()
    print("Confidence")
    print("-" * 100)
    print(
        "Score :",
        round(
            response.confidence.confidence_score,
            6,
        ),
    )
    print(
        "Top1 reranker :",
        round(
            response.confidence
            .features
            .top1_reranker_score,
            6,
        ),
    )
    print(
        "RRF :",
        response.confidence
        .features
        .top1_rrf_score,
    )
    print(
        "Failed rules :",
        response.confidence.failed_rules,
    )

    print()
    print("Citations")
    print("-" * 100)

    if response.citations:
        for citation in response.citations:
            print(citation)
    else:
        print("Aucune citation")

    print()
    print("Sources utilisées")
    print("-" * 100)

    for source in response.sources:
        print(
            f"{source.rank}. {source.source}"
        )
        print(
            f"   Chunk : {source.chunk_id}"
        )
        print(
            f"   Reranker : "
            f"{source.reranker_score:.6f}"
        )
        print(
            f"   RRF : {source.rrf_score:.8f}"
        )
        print()

    print("Utilisation LLM")
    print("-" * 100)

    if response.generation is None:
        print(
            "Aucun appel de génération effectué."
        )

    else:
        print(
            "Input tokens :",
            response
            .generation
            .usage
            .input_tokens,
        )
        print(
            "Output tokens :",
            response
            .generation
            .usage
            .output_tokens,
        )
        print(
            "Total tokens :",
            response
            .generation
            .usage
            .total_tokens,
        )
        print(
            "Temps fournisseur :",
            response
            .generation
            .generation_time_ms,
            "ms",
        )

    print()
    print("Validation des citations")
    print("-" * 100)

    if response.citation_validation is None:
        print(
            "Validation non exécutée."
        )

    else:
        for key, value in (
            response
            .citation_validation
            .to_dict()
            .items()
        ):
            print(
                f"{key} : {value}"
            )

    print()
    print("Timings du pipeline")
    print("-" * 100)

    for key, value in (
        response
        .timings
        .to_dict()
        .items()
    ):
        print(
            f"{key} : {value}"
        )

    print()
    print(
        "Refusal reason :",
        response.refusal_reason,
    )


def validate_answerable_response(
    response: RAGResponse,
) -> None:
    """
    Vérifie que la question Azure a déclenché
    une génération Hugging Face valide.
    """

    if not response.accepted:
        raise RuntimeError(
            "La question Azure a été refusée."
        )

    if response.provider != "huggingface":
        raise RuntimeError(
            "Le fournisseur attendu est "
            "huggingface."
        )

    if response.model_name != (
        "google/gemma-2-2b-it"
    ):
        raise RuntimeError(
            "Le modèle Hugging Face retourné "
            "est incorrect."
        )

    if response.generation is None:
        raise RuntimeError(
            "Aucune génération n'a été exécutée "
            "pour la question répondable."
        )

    if not response.citations:
        raise RuntimeError(
            "La réponse ne contient aucune "
            "citation valide."
        )

    if (
        response.citation_validation
        is None
        or not response
        .citation_validation
        .valid
    ):
        raise RuntimeError(
            "La validation des citations "
            "a échoué."
        )


def validate_rejected_response(
    response: RAGResponse,
) -> None:
    """
    Vérifie que la question hors corpus a été
    refusée avant la génération.
    """

    if response.accepted:
        raise RuntimeError(
            "La question Cisco IOS a été "
            "acceptée à tort."
        )

    if response.generation is not None:
        raise RuntimeError(
            "Un appel LLM a été effectué pour "
            "une question refusée."
        )

    if response.provider is not None:
        raise RuntimeError(
            "Le provider doit être None pour "
            "une question refusée avant génération."
        )

    if response.citations:
        raise RuntimeError(
            "Une question refusée ne doit pas "
            "contenir de citation."
        )

    if (
        "top1_score"
        not in response
        .confidence
        .failed_rules
    ):
        raise RuntimeError(
            "La question hors corpus devait "
            "échouer sur le score Top1."
        )


def main() -> None:
    print("=" * 100)
    print(
        "TEST D'INTÉGRATION DU RAG "
        "AVEC HUGGING FACE"
    )
    print("=" * 100)

    rag = RAGPipeline.from_settings(
        language="en"
    )

    # Une seule tentative distante en cas d'erreur,
    # afin d'éviter de consommer plusieurs appels.
    rag.llm_manager.max_attempts_per_provider = 1
    rag.llm_manager.retry_delay_seconds = 0.0

    print(
        "Provider principal :",
        rag.llm_manager.provider_name,
    )
    print(
        "Modèle principal :",
        rag.llm_manager.model_name,
    )
    print(
        "Health :",
        rag.llm_manager.health_check(),
    )

    try:
        # ====================================================
        # Test 1 : question répondable
        # ====================================================

        print()
        print("=" * 100)
        print("TEST 1 — QUESTION AZURE RÉPONDABLE")
        print("=" * 100)
        print(ANSWERABLE_QUESTION)

        answerable_response = rag.answer(
            ANSWERABLE_QUESTION
        )

        print_response(
            answerable_response
        )

        validate_answerable_response(
            answerable_response
        )

        answerable_attempts = (
            rag.llm_manager.last_attempts
        )

        if not answerable_attempts:
            raise RuntimeError(
                "Aucune tentative LLM enregistrée."
            )

        print()
        print("Tentatives LLM")
        print("-" * 100)

        for attempt in answerable_attempts:
            print(
                attempt.to_dict()
            )

        # ====================================================
        # Test 2 : question technique hors corpus
        # ====================================================

        print()
        print("=" * 100)
        print(
            "TEST 2 — QUESTION TECHNIQUE "
            "HORS CORPUS"
        )
        print("=" * 100)
        print(OUT_OF_CORPUS_QUESTION)

        attempts_before_rejection = (
            rag.llm_manager.last_attempts
        )

        rejected_response = rag.answer(
            OUT_OF_CORPUS_QUESTION
        )

        print_response(
            rejected_response
        )

        validate_rejected_response(
            rejected_response
        )

        attempts_after_rejection = (
            rag.llm_manager.last_attempts
        )

        # Le manager n'est pas appelé lorsque le
        # Confidence Gate refuse la question.
        if (
            attempts_after_rejection
            != attempts_before_rejection
        ):
            raise RuntimeError(
                "L'historique du LLMManager a changé : "
                "un appel distant a peut-être été effectué "
                "pour la question refusée."
            )

        print()
        print("=" * 100)
        print(
            "TEST D'INTÉGRATION HUGGING FACE RÉUSSI"
        )
        print("=" * 100)
        print(
            "Question Azure : génération effectuée."
        )
        print(
            "Question Cisco IOS : refusée sans "
            "appel Hugging Face."
        )

    finally:
        rag.unload_models()


if __name__ == "__main__":
    main()