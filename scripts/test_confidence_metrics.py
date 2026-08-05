from __future__ import annotations

from src.confidence import (
    calculate_confidence_metrics,
)


def main() -> None:
    expected = [
        True,
        True,
        True,
        False,
        False,
        False,
    ]

    predicted = [
        True,
        True,
        False,
        False,
        True,
        False,
    ]

    metrics = calculate_confidence_metrics(
        expected_labels=expected,
        predicted_labels=predicted,
    )

    print("=" * 70)
    print("TEST DES MÉTRIQUES DU CONFIDENCE GATE")
    print("=" * 70)

    for key, value in metrics.to_dict().items():
        print(f"{key} : {value}")


if __name__ == "__main__":
    main()