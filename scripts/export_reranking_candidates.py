from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.retrieval_metrics import (
    find_evidence_rank,
    find_source_rank,
    load_evaluation_questions,
)
from src.retrieval.hybrid_rrf_retriever import (
    HybridRRFRetriever,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_30_questions.csv"
)

QUERY_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "e5_base_query_embeddings.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmarking"
    / "04_reranking_benchmark"
)

CANDIDATES_PATH = (
    OUTPUT_DIR
    / "hybrid_top20_candidates.jsonl"
)

BASELINE_PATH = (
    OUTPUT_DIR
    / "hybrid_top20_baseline.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "reranking_dataset_manifest.json"
)

EXPECTED_QUESTIONS = 30
EXPECTED_DIMENSION = 768
EXPECTED_CHUNKS = 1478

RETRIEVER_CANDIDATE_K = 50
RERANK_CANDIDATE_K = 20
RRF_CONSTANT = 60


# ============================================================
# Fonctions utilitaires
# ============================================================

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def load_embeddings(path: Path) -> np.ndarray:
    embeddings = np.load(
        path,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    expected_shape = (
        EXPECTED_QUESTIONS,
        EXPECTED_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Forme attendue : {expected_shape}. "
            f"Forme trouvée : {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs "
            "NaN ou infinies."
        )

    return embeddings


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        raise ValueError(
            "Aucun enregistrement à sauvegarder."
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# Programme principal
# ============================================================

def main() -> None:
    print("=" * 100)
    print("EXPORT DES CANDIDATS POUR LE BENCHMARK DE RERANKING")
    print("=" * 100)

    require_file(EVALUATION_PATH)
    require_file(QUERY_EMBEDDINGS_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_evaluation_questions(
        EVALUATION_PATH
    )

    embeddings = load_embeddings(
        QUERY_EMBEDDINGS_PATH
    )

    if len(questions) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"{EXPECTED_QUESTIONS} questions attendues, "
            f"{len(questions)} trouvées."
        )

    retriever = HybridRRFRetriever(
        candidate_k=RETRIEVER_CANDIDATE_K,
        final_top_k=RERANK_CANDIDATE_K,
        rrf_constant=RRF_CONSTANT,
    )

    vector_count = retriever.count_vector_chunks()
    bm25_count = retriever.count_bm25_chunks()

    if vector_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"pgvector contient {vector_count} chunks "
            f"au lieu de {EXPECTED_CHUNKS}."
        )

    if bm25_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"BM25 contient {bm25_count} chunks "
            f"au lieu de {EXPECTED_CHUNKS}."
        )

    baseline_records: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    with CANDIDATES_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, (
            question,
            embedding,
        ) in enumerate(
            zip(questions, embeddings),
            start=1,
        ):
            query_start = time.perf_counter()

            results = retriever.search(
                question=question["question"],
                query_embedding=embedding,
                candidate_k=RETRIEVER_CANDIDATE_K,
                final_top_k=RERANK_CANDIDATE_K,
            )

            retrieval_time_ms = (
                time.perf_counter()
                - query_start
            ) * 1000

            source_rank = find_source_rank(
                results,
                question["expected_source"],
            )

            evidence_rank = find_evidence_rank(
                results,
                question["gold_evidence"],
            )

            output_record = {
                "id": question["id"],
                "question": question["question"],
                "answer": question.get("answer", ""),
                "format": question.get(
                    "format",
                    question.get(
                        "document_format",
                        "",
                    ),
                ),
                "expected_source": (
                    question["expected_source"]
                ),
                "gold_evidence": (
                    question["gold_evidence"]
                ),
                "retrieval_time_ms": round(
                    retrieval_time_ms,
                    4,
                ),
                "candidates": [],
            }

            for result in results:
                output_record["candidates"].append(
                    {
                        "hybrid_rank": result.rank,
                        "chunk_position": (
                            result.chunk_position
                        ),
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "source": result.source,
                        "document_format": (
                            result.document_format
                        ),
                        "page_number": (
                            result.page_number
                        ),
                        "rrf_score": round(
                            result.rrf_score,
                            10,
                        ),
                        "vector_rank": (
                            result.vector_rank
                        ),
                        "bm25_rank": (
                            result.bm25_rank
                        ),
                        "cosine_similarity": (
                            result.cosine_similarity
                        ),
                        "bm25_score": (
                            result.bm25_score
                        ),
                    }
                )

            output_file.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            baseline_records.append(
                {
                    "id": question["id"],
                    "source_rank_top20": (
                        source_rank
                        if source_rank is not None
                        else ""
                    ),
                    "evidence_rank_top20": (
                        evidence_rank
                        if evidence_rank is not None
                        else ""
                    ),
                    "retrieval_time_ms": round(
                        retrieval_time_ms,
                        4,
                    ),
                    "number_of_candidates": len(
                        results
                    ),
                }
            )

            print(
                f"[{index:02d}/{EXPECTED_QUESTIONS}] "
                f"{question['id']} | "
                f"Evidence rank Top20: "
                f"{evidence_rank or 'absente'} | "
                f"{retrieval_time_ms:.2f} ms"
            )

    write_csv(
        BASELINE_PATH,
        baseline_records,
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    manifest = {
        "benchmark": "cross_encoder_reranking",
        "number_of_questions": (
            EXPECTED_QUESTIONS
        ),
        "number_of_indexed_chunks": (
            EXPECTED_CHUNKS
        ),
        "retrieval_method": "hybrid_rrf",
        "vector_candidate_k": (
            RETRIEVER_CANDIDATE_K
        ),
        "bm25_candidate_k": (
            RETRIEVER_CANDIDATE_K
        ),
        "reranking_candidate_k": (
            RERANK_CANDIDATE_K
        ),
        "rrf_constant": RRF_CONSTANT,
        "total_export_time_seconds": round(
            total_time,
            4,
        ),
        "files": {
            "candidates": str(
                CANDIDATES_PATH
            ),
            "baseline": str(
                BASELINE_PATH
            ),
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print("EXPORT TERMINÉ")
    print("=" * 100)
    print("Candidats :", CANDIDATES_PATH)
    print("Baseline :", BASELINE_PATH)
    print("Manifeste :", MANIFEST_PATH)
    print(
        "Temps total :",
        round(total_time, 3),
        "secondes",
    )


if __name__ == "__main__":
    main()