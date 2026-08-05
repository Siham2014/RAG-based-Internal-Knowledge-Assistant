from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceFeatures:
    """
    Signaux extraits des résultats rerankés.

    Ces indicateurs seront utilisés par le Confidence Gate
    pour décider si les passages récupérés sont suffisamment
    pertinents pour autoriser la génération d'une réponse.
    """

    number_of_results: int

    top1_reranker_score: float
    top2_reranker_score: float | None
    reranker_margin: float

    top1_cosine_similarity: float | None
    top1_rrf_score: float | None

    mean_top_k_reranker_score: float
    supporting_results_count: int

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit les caractéristiques en dictionnaire.
        """

        return {
            "number_of_results": (
                self.number_of_results
            ),
            "top1_reranker_score": (
                self.top1_reranker_score
            ),
            "top2_reranker_score": (
                self.top2_reranker_score
            ),
            "reranker_margin": (
                self.reranker_margin
            ),
            "top1_cosine_similarity": (
                self.top1_cosine_similarity
            ),
            "top1_rrf_score": (
                self.top1_rrf_score
            ),
            "mean_top_k_reranker_score": (
                self.mean_top_k_reranker_score
            ),
            "supporting_results_count": (
                self.supporting_results_count
            ),
        }


@dataclass(frozen=True)
class ConfidenceThresholds:
    """
    Seuils utilisés pour accepter ou refuser une question.

    Ces valeurs seront choisies expérimentalement pendant
    le benchmark du Confidence Gate.
    """

    minimum_top1_score: float
    minimum_margin: float
    minimum_cosine_similarity: float | None = None
    minimum_rrf_score: float | None = None
    minimum_supporting_results: int = 1

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit les seuils en dictionnaire.
        """

        return {
            "minimum_top1_score": (
                self.minimum_top1_score
            ),
            "minimum_margin": (
                self.minimum_margin
            ),
            "minimum_cosine_similarity": (
                self.minimum_cosine_similarity
            ),
            "minimum_rrf_score": (
                self.minimum_rrf_score
            ),
            "minimum_supporting_results": (
                self.minimum_supporting_results
            ),
        }


@dataclass(frozen=True)
class ConfidenceDecision:
    """
    Décision produite par le Confidence Gate.
    """

    accepted: bool
    confidence_score: float
    reason: str

    features: ConfidenceFeatures
    thresholds: ConfidenceThresholds

    failed_rules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit la décision en dictionnaire sérialisable.
        """

        return {
            "accepted": self.accepted,
            "confidence_score": (
                self.confidence_score
            ),
            "reason": self.reason,
            "failed_rules": list(
                self.failed_rules
            ),
            "features": (
                self.features.to_dict()
            ),
            "thresholds": (
                self.thresholds.to_dict()
            ),
        }


@dataclass(frozen=True)
class ConfidenceEvaluationRecord:
    """
    Enregistrement utilisé pour le benchmark scientifique.

    Une ligne correspond à une question évaluée.
    """

    question_id: str
    question: str

    expected_answerable: bool
    predicted_answerable: bool

    confidence_score: float

    top1_reranker_score: float
    top2_reranker_score: float | None
    reranker_margin: float

    top1_cosine_similarity: float | None
    top1_rrf_score: float | None

    supporting_results_count: int
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit l'enregistrement en dictionnaire.
        """

        return {
            "question_id": (
                self.question_id
            ),
            "question": self.question,
            "expected_answerable": (
                self.expected_answerable
            ),
            "predicted_answerable": (
                self.predicted_answerable
            ),
            "confidence_score": (
                self.confidence_score
            ),
            "top1_reranker_score": (
                self.top1_reranker_score
            ),
            "top2_reranker_score": (
                self.top2_reranker_score
            ),
            "reranker_margin": (
                self.reranker_margin
            ),
            "top1_cosine_similarity": (
                self.top1_cosine_similarity
            ),
            "top1_rrf_score": (
                self.top1_rrf_score
            ),
            "supporting_results_count": (
                self.supporting_results_count
            ),
            "decision_reason": (
                self.decision_reason
            ),
        }