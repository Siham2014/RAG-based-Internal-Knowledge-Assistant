from __future__ import annotations

from statistics import mean

from src.confidence.models import (
    ConfidenceDecision,
    ConfidenceFeatures,
    ConfidenceThresholds,
)
from src.reranking import (
    RerankedSearchResult,
)


class ConfidenceGate:
    """
    Confidence Gate basé sur des règles.

    Cette première version ne dépend
    d'aucun modèle ML.

    Les seuils seront optimisés
    expérimentalement pendant
    le benchmark.
    """

    def __init__(
        self,
        thresholds: ConfidenceThresholds,
    ) -> None:

        self.thresholds = thresholds

    ###########################################################
    # Extraction des caractéristiques
    ###########################################################

    def extract_features(
        self,
        results: list[RerankedSearchResult],
    ) -> ConfidenceFeatures:

        if len(results) == 0:

            return ConfidenceFeatures(

                number_of_results=0,

                top1_reranker_score=0.0,
                top2_reranker_score=None,
                reranker_margin=0.0,

                top1_cosine_similarity=None,
                top1_rrf_score=None,

                mean_top_k_reranker_score=0.0,

                supporting_results_count=0,
            )

        top1 = results[0]

        top2 = (
            results[1]
            if len(results) > 1
            else None
        )

        top1_score = (
            float(top1.reranker_score)
        )

        top2_score = (
            float(top2.reranker_score)
            if top2 is not None
            else None
        )

        margin = (
            top1_score - top2_score
            if top2_score is not None
            else top1_score
        )

        scores = [
            float(r.reranker_score)
            for r in results
        ]

        supporting = sum(
            1
            for score in scores
            if score >= 0.50
        )

        return ConfidenceFeatures(

            number_of_results=len(results),

            top1_reranker_score=top1_score,

            top2_reranker_score=top2_score,

            reranker_margin=margin,

            top1_cosine_similarity=(
                top1.cosine_similarity
            ),

            top1_rrf_score=(
                top1.rrf_score
            ),

            mean_top_k_reranker_score=(
                mean(scores)
            ),

            supporting_results_count=(
                supporting
            ),
        )

    ###########################################################
    # Décision
    ###########################################################

    def evaluate(
        self,
        results: list[RerankedSearchResult],
    ) -> ConfidenceDecision:

        features = (
            self.extract_features(
                results
            )
        )

        failed_rules = []

        if (
            features.top1_reranker_score
            <
            self.thresholds.minimum_top1_score
        ):

            failed_rules.append(
                "top1_score"
            )

        if (
            features.reranker_margin
            <
            self.thresholds.minimum_margin
        ):

            failed_rules.append(
                "margin"
            )

        if (
            self.thresholds.minimum_cosine_similarity
            is not None
        ):

            cosine = (
                features
                .top1_cosine_similarity
                or 0.0
            )

            if (
                cosine
                <
                self.thresholds
                .minimum_cosine_similarity
            ):

                failed_rules.append(
                    "cosine"
                )

        if (
            self.thresholds.minimum_rrf_score
            is not None
        ):

            rrf = (
                features.top1_rrf_score
                or 0.0
            )

            if (
                rrf
                <
                self.thresholds
                .minimum_rrf_score
            ):

                failed_rules.append(
                    "rrf"
                )

        if (
            features.supporting_results_count
            <
            self.thresholds
            .minimum_supporting_results
        ):

            failed_rules.append(
                "supporting_results"
            )

        accepted = (
            len(failed_rules)
            == 0
        )

        confidence = (
            features.top1_reranker_score
        )

        reason = (
            "Accepted"
            if accepted
            else "Rejected"
        )

        return ConfidenceDecision(

            accepted=accepted,

            confidence_score=confidence,

            reason=reason,

            features=features,

            thresholds=self.thresholds,

            failed_rules=tuple(
                failed_rules
            ),
        )