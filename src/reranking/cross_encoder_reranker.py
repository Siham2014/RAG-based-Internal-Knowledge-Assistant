from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_rrf_retriever import (
    HybridSearchResult,
)


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RerankedSearchResult:
    """
    Résultat produit après reranking d'un candidat Hybrid RRF.

    Le résultat conserve les scores et les rangs issus
    des étapes vectorielle, lexicale et hybride afin de
    pouvoir être utilisé ensuite par le Confidence Gate.
    """

    rank: int
    hybrid_rank: int

    chunk_position: int
    chunk_id: str
    content: str
    source: str
    document_format: str
    page_number: int | None
    metadata: dict

    reranker_score: float
    rrf_score: float

    vector_rank: int | None
    bm25_rank: int | None

    cosine_similarity: float | None
    cosine_distance: float | None
    bm25_score: float | None


class CrossEncoderReranker:
    """
    Réordonne les candidats d'un retriever hybride à l'aide
    d'un modèle Cross-Encoder configurable.

    Exemple :
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-base"
        )

        results = reranker.rerank(
            question="What is Azure?",
            candidates=hybrid_results,
            top_k=5,
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:

        normalized_model_name = str(
            model_name
        ).strip()

        if not normalized_model_name:
            raise ValueError(
                "model_name ne peut pas être vide."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size doit être supérieur à zéro."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length doit être supérieur à zéro."
            )

        self.model_name = normalized_model_name
        self.device = self._resolve_device(
            device
        )
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self._model: CrossEncoder | None = None

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> str:
        """
        Sélectionne le périphérique d'exécution.

        Valeurs acceptées :
        - None ou "auto"
        - "cpu"
        - "cuda"
        """

        if requested_device is None:
            requested_device = "auto"

        normalized_device = (
            str(requested_device)
            .strip()
            .lower()
        )

        if normalized_device == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if normalized_device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device doit être auto, cpu ou cuda."
            )

        if (
            normalized_device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA a été demandé, mais aucun GPU "
                "compatible n'est disponible."
            )

        return normalized_device

    @property
    def is_loaded(self) -> bool:
        """
        Indique si le modèle a déjà été chargé.
        """

        return self._model is not None

    def load_model(self) -> None:
        """
        Charge le Cross-Encoder une seule fois.

        Le chargement est différé afin d'éviter de télécharger
        ou charger le modèle dès l'import du module.
        """

        if self._model is not None:
            return

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )

    def unload_model(self) -> None:
        """
        Libère le modèle et la mémoire GPU si nécessaire.
        """

        self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        normalized_question = str(
            question
        ).strip()

        if not normalized_question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        return normalized_question

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[
            HybridSearchResult
        ],
    ) -> list[HybridSearchResult]:
        candidate_list = list(candidates)

        if not candidate_list:
            raise ValueError(
                "La liste des candidats ne peut pas être vide."
            )

        for index, candidate in enumerate(
            candidate_list,
            start=1,
        ):
            if not isinstance(
                candidate,
                HybridSearchResult,
            ):
                raise TypeError(
                    "Tous les candidats doivent être des "
                    "instances de HybridSearchResult. "
                    f"Élément invalide au rang {index}."
                )

            if not candidate.content.strip():
                raise ValueError(
                    f"Le candidat {candidate.chunk_id} "
                    "contient un texte vide."
                )

        return candidate_list

    @staticmethod
    def _validate_top_k(
        top_k: int,
        number_of_candidates: int,
    ) -> int:
        normalized_top_k = int(top_k)

        if normalized_top_k <= 0:
            raise ValueError(
                "top_k doit être supérieur à zéro."
            )

        return min(
            normalized_top_k,
            number_of_candidates,
        )

    def score_candidates(
        self,
        question: str,
        candidates: Sequence[
            HybridSearchResult
        ],
    ) -> np.ndarray:
        """
        Calcule un score Cross-Encoder pour chaque candidat.

        L'ordre des scores correspond à l'ordre des candidats
        fournis en entrée.
        """

        normalized_question = (
            self._validate_question(
                question
            )
        )

        candidate_list = (
            self._validate_candidates(
                candidates
            )
        )

        self.load_model()

        if self._model is None:
            raise RuntimeError(
                "Le modèle de reranking n'a pas été chargé."
            )

        sentence_pairs = [
            [
                normalized_question,
                candidate.content,
            ]
            for candidate in candidate_list
        ]

        scores = self._model.predict(
            sentence_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores_array = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if scores_array.shape != (
            len(candidate_list),
        ):
            raise RuntimeError(
                "Le modèle a retourné un nombre de scores "
                "incompatible avec le nombre de candidats."
            )

        if not np.isfinite(
            scores_array
        ).all():
            raise RuntimeError(
                "Le modèle a produit des scores NaN "
                "ou infinis."
            )

        return scores_array

    def rerank(
        self,
        question: str,
        candidates: Sequence[
            HybridSearchResult
        ],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RerankedSearchResult]:
        """
        Réordonne les candidats selon leur score Cross-Encoder.

        Args:
            question:
                Question originale de l'utilisateur.

            candidates:
                Résultats issus du HybridRRFRetriever.

            top_k:
                Nombre de résultats finaux à retourner.

        Returns:
            Liste de RerankedSearchResult triée par score
            décroissant.
        """

        candidate_list = (
            self._validate_candidates(
                candidates
            )
        )

        final_limit = self._validate_top_k(
            top_k=top_k,
            number_of_candidates=len(
                candidate_list
            ),
        )

        scores = self.score_candidates(
            question=question,
            candidates=candidate_list,
        )

        scored_candidates = list(
            zip(
                candidate_list,
                scores,
            )
        )

        scored_candidates.sort(
            key=lambda item: (
                -float(item[1]),
                item[0].rank,
                item[0].chunk_id,
            )
        )

        reranked_results: list[
            RerankedSearchResult
        ] = []

        for final_rank, (
            candidate,
            score,
        ) in enumerate(
            scored_candidates[
                :final_limit
            ],
            start=1,
        ):
            reranked_results.append(
                RerankedSearchResult(
                    rank=final_rank,
                    hybrid_rank=candidate.rank,

                    chunk_position=(
                        candidate.chunk_position
                    ),
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    source=candidate.source,
                    document_format=(
                        candidate.document_format
                    ),
                    page_number=(
                        candidate.page_number
                    ),
                    metadata=dict(
                        candidate.metadata
                    ),

                    reranker_score=float(
                        score
                    ),
                    rrf_score=float(
                        candidate.rrf_score
                    ),

                    vector_rank=(
                        candidate.vector_rank
                    ),
                    bm25_rank=(
                        candidate.bm25_rank
                    ),

                    cosine_similarity=(
                        candidate.cosine_similarity
                    ),
                    cosine_distance=(
                        candidate.cosine_distance
                    ),
                    bm25_score=(
                        candidate.bm25_score
                    ),
                )
            )

        return reranked_results