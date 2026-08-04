from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25SearchResult,
)
from src.retrieval.vector_retriever import (
    PgVectorRetriever,
    VectorSearchResult,
)


DEFAULT_FINAL_TOP_K = 10
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_CONSTANT = 60


@dataclass(frozen=True)
class HybridSearchResult:
    """
    Résultat final produit après fusion RRF.
    """

    rank: int
    chunk_position: int
    chunk_id: str
    content: str
    source: str
    document_format: str
    page_number: int | None
    metadata: dict[str, Any]

    rrf_score: float

    vector_rank: int | None
    bm25_rank: int | None

    cosine_similarity: float | None
    cosine_distance: float | None
    bm25_score: float | None


class HybridRRFRetriever:
    """
    Fusionne les résultats de pgvector et de BM25
    avec Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_retriever: PgVectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ) -> None:

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k doit être supérieur à zéro."
            )

        if final_top_k <= 0:
            raise ValueError(
                "final_top_k doit être supérieur à zéro."
            )

        if final_top_k > candidate_k * 2:
            raise ValueError(
                "final_top_k est trop grand par rapport "
                "au nombre total de candidats."
            )

        if rrf_constant < 0:
            raise ValueError(
                "rrf_constant doit être positif ou nul."
            )

        self.vector_retriever = (
            vector_retriever
            if vector_retriever is not None
            else PgVectorRetriever(
                default_top_k=candidate_k
            )
        )

        self.bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever(
                default_top_k=candidate_k
            )
        )

        self.candidate_k = candidate_k
        self.final_top_k = final_top_k
        self.rrf_constant = rrf_constant

    def count_vector_chunks(self) -> int:
        """
        Nombre de chunks présents dans pgvector.
        """

        return (
            self.vector_retriever
            .count_indexed_chunks()
        )

    def count_bm25_chunks(self) -> int:
        """
        Nombre de chunks présents dans SQLite FTS5.
        """

        return (
            self.bm25_retriever
            .count_indexed_chunks()
        )

    def _rrf_contribution(
        self,
        rank: int,
    ) -> float:
        """
        Calcule la contribution RRF d'un rang.
        """

        return 1.0 / (
            self.rrf_constant
            + rank
        )

    def search(
        self,
        question: str,
        query_embedding: np.ndarray,
        final_top_k: int | None = None,
        candidate_k: int | None = None,
        document_format: str | None = None,
    ) -> list[HybridSearchResult]:
        """
        Exécute pgvector et BM25 puis fusionne leurs
        classements avec RRF.
        """

        final_limit = (
            self.final_top_k
            if final_top_k is None
            else int(final_top_k)
        )

        candidate_limit = (
            self.candidate_k
            if candidate_k is None
            else int(candidate_k)
        )

        if final_limit <= 0:
            raise ValueError(
                "final_top_k doit être supérieur à zéro."
            )

        if candidate_limit <= 0:
            raise ValueError(
                "candidate_k doit être supérieur à zéro."
            )

        vector_results = (
            self.vector_retriever
            .search_by_embedding(
                query_embedding=query_embedding,
                top_k=candidate_limit,
                document_format=document_format,
            )
        )

        bm25_results = (
            self.bm25_retriever
            .search(
                question=question,
                top_k=candidate_limit,
                document_format=document_format,
            )
        )

        combined: dict[
            str,
            dict[str, Any],
        ] = {}

        self._add_vector_results(
            combined=combined,
            results=vector_results,
        )

        self._add_bm25_results(
            combined=combined,
            results=bm25_results,
        )

        ordered_candidates = sorted(
            combined.values(),
            key=lambda item: (
                -float(item["rrf_score"]),
                min(
                    item["vector_rank"]
                    if item["vector_rank"] is not None
                    else 10**9,
                    item["bm25_rank"]
                    if item["bm25_rank"] is not None
                    else 10**9,
                ),
                item["chunk_id"],
            ),
        )

        final_results: list[
            HybridSearchResult
        ] = []

        for final_rank, item in enumerate(
            ordered_candidates[
                :final_limit
            ],
            start=1,
        ):

            final_results.append(
                HybridSearchResult(
                    rank=final_rank,
                    chunk_position=int(
                        item["chunk_position"]
                    ),
                    chunk_id=str(
                        item["chunk_id"]
                    ),
                    content=str(
                        item["content"]
                    ),
                    source=str(
                        item["source"]
                    ),
                    document_format=str(
                        item["document_format"]
                    ),
                    page_number=(
                        int(item["page_number"])
                        if item["page_number"]
                        is not None
                        else None
                    ),
                    metadata=dict(
                        item["metadata"]
                    ),
                    rrf_score=float(
                        item["rrf_score"]
                    ),
                    vector_rank=(
                        int(item["vector_rank"])
                        if item["vector_rank"]
                        is not None
                        else None
                    ),
                    bm25_rank=(
                        int(item["bm25_rank"])
                        if item["bm25_rank"]
                        is not None
                        else None
                    ),
                    cosine_similarity=(
                        float(
                            item[
                                "cosine_similarity"
                            ]
                        )
                        if item[
                            "cosine_similarity"
                        ] is not None
                        else None
                    ),
                    cosine_distance=(
                        float(
                            item[
                                "cosine_distance"
                            ]
                        )
                        if item[
                            "cosine_distance"
                        ] is not None
                        else None
                    ),
                    bm25_score=(
                        float(
                            item["bm25_score"]
                        )
                        if item["bm25_score"]
                        is not None
                        else None
                    ),
                )
            )

        return final_results

    def _add_vector_results(
        self,
        combined: dict[
            str,
            dict[str, Any],
        ],
        results: list[
            VectorSearchResult
        ],
    ) -> None:
        """
        Ajoute les candidats issus de pgvector.
        """

        for result in results:

            item = combined.setdefault(
                result.chunk_id,
                {
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
                    "metadata": (
                        result.metadata
                    ),
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "cosine_similarity": None,
                    "cosine_distance": None,
                    "bm25_score": None,
                },
            )

            item["vector_rank"] = (
                result.rank
            )

            item["cosine_similarity"] = (
                result.cosine_similarity
            )

            item["cosine_distance"] = (
                result.cosine_distance
            )

            item["rrf_score"] += (
                self._rrf_contribution(
                    result.rank
                )
            )

    def _add_bm25_results(
        self,
        combined: dict[
            str,
            dict[str, Any],
        ],
        results: list[
            BM25SearchResult
        ],
    ) -> None:
        """
        Ajoute les candidats issus de BM25.
        """

        for result in results:

            item = combined.setdefault(
                result.chunk_id,
                {
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
                    "metadata": {},
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "bm25_rank": None,
                    "cosine_similarity": None,
                    "cosine_distance": None,
                    "bm25_score": None,
                },
            )

            item["bm25_rank"] = (
                result.rank
            )

            item["bm25_score"] = (
                result.bm25_score
            )

            item["rrf_score"] += (
                self._rrf_contribution(
                    result.rank
                )
            )