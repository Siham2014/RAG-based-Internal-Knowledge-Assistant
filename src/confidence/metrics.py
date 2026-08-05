from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ConfidenceMetrics:
    """
    Métriques binaires du Confidence Gate.

    Classe positive :
        question répondable / acceptée.

    Classe négative :
        question non répondable / refusée.
    """

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float

    false_acceptance_rate: float
    false_rejection_rate: float

    coverage: float
    rejection_rate: float

    number_of_questions: int
    number_of_accepted_questions: int
    number_of_rejected_questions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "false_acceptance_rate": (
                self.false_acceptance_rate
            ),
            "false_rejection_rate": (
                self.false_rejection_rate
            ),
            "coverage": self.coverage,
            "rejection_rate": self.rejection_rate,
            "number_of_questions": (
                self.number_of_questions
            ),
            "number_of_accepted_questions": (
                self.number_of_accepted_questions
            ),
            "number_of_rejected_questions": (
                self.number_of_rejected_questions
            ),
        }


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    Division protégée contre une division par zéro.
    """

    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def calculate_confusion_matrix(
    expected_labels: Iterable[bool],
    predicted_labels: Iterable[bool],
) -> tuple[int, int, int, int]:
    """
    Calcule TP, TN, FP et FN.

    TP :
        question répondable correctement acceptée.

    TN :
        question non répondable correctement refusée.

    FP :
        question non répondable incorrectement acceptée.

    FN :
        question répondable incorrectement refusée.
    """

    expected_list = list(expected_labels)
    predicted_list = list(predicted_labels)

    if len(expected_list) != len(predicted_list):
        raise ValueError(
            "Le nombre de labels attendus doit être égal "
            "au nombre de prédictions."
        )

    if not expected_list:
        raise ValueError(
            "Aucun label fourni pour le calcul des métriques."
        )

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for expected, predicted in zip(
        expected_list,
        predicted_list,
    ):
        if not isinstance(expected, bool):
            raise TypeError(
                "Tous les labels attendus doivent être "
                "des booléens."
            )

        if not isinstance(predicted, bool):
            raise TypeError(
                "Toutes les prédictions doivent être "
                "des booléens."
            )

        if expected and predicted:
            true_positive += 1

        elif not expected and not predicted:
            true_negative += 1

        elif not expected and predicted:
            false_positive += 1

        else:
            false_negative += 1

    return (
        true_positive,
        true_negative,
        false_positive,
        false_negative,
    )


def calculate_confidence_metrics(
    expected_labels: Iterable[bool],
    predicted_labels: Iterable[bool],
    digits: int = 6,
) -> ConfidenceMetrics:
    """
    Calcule toutes les métriques du Confidence Gate.
    """

    expected_list = list(expected_labels)
    predicted_list = list(predicted_labels)

    (
        true_positive,
        true_negative,
        false_positive,
        false_negative,
    ) = calculate_confusion_matrix(
        expected_labels=expected_list,
        predicted_labels=predicted_list,
    )

    number_of_questions = len(
        expected_list
    )

    number_of_accepted_questions = sum(
        1
        for predicted in predicted_list
        if predicted
    )

    number_of_rejected_questions = (
        number_of_questions
        - number_of_accepted_questions
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        number_of_questions,
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    # FAR : proportion de questions non répondables
    # incorrectement acceptées.
    false_acceptance_rate = safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    # FRR : proportion de questions répondables
    # incorrectement refusées.
    false_rejection_rate = safe_divide(
        false_negative,
        false_negative + true_positive,
    )

    # Coverage : proportion de questions acceptées.
    coverage = safe_divide(
        number_of_accepted_questions,
        number_of_questions,
    )

    rejection_rate = safe_divide(
        number_of_rejected_questions,
        number_of_questions,
    )

    return ConfidenceMetrics(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,

        accuracy=round(
            accuracy,
            digits,
        ),
        precision=round(
            precision,
            digits,
        ),
        recall=round(
            recall,
            digits,
        ),
        specificity=round(
            specificity,
            digits,
        ),
        f1_score=round(
            f1_score,
            digits,
        ),

        false_acceptance_rate=round(
            false_acceptance_rate,
            digits,
        ),
        false_rejection_rate=round(
            false_rejection_rate,
            digits,
        ),

        coverage=round(
            coverage,
            digits,
        ),
        rejection_rate=round(
            rejection_rate,
            digits,
        ),

        number_of_questions=(
            number_of_questions
        ),
        number_of_accepted_questions=(
            number_of_accepted_questions
        ),
        number_of_rejected_questions=(
            number_of_rejected_questions
        ),
    )