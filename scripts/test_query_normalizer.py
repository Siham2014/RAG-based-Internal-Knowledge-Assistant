from __future__ import annotations

from src.query_processing.query_normalizer import (
    QueryNormalizer,
)


QUESTIONS = [
    "what is clod computin",
    "what is claud compoting",
    "How cn I improve Azure reliabiliti?",
    "What is Azure Databrikc?",
    "explan kubirnitis archtectur",
    "What is AKS?",
    "Explain cloud scalabiliti",
    "How to imprauve reliability",
]


def main() -> None:
    normalizer = QueryNormalizer()

    print("=" * 100)
    print("QUERY NORMALIZATION TEST")
    print("=" * 100)

    for question in QUESTIONS:
        normalized = normalizer.normalize(
            question
        )

        print()
        print("ORIGINAL   :", question)
        print("NORMALIZED :", normalized)
        print("-" * 100)


if __name__ == "__main__":
    main()