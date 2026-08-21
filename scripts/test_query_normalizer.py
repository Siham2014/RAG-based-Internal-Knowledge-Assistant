from __future__ import annotations

from src.query_processing.query_normalizer import (
    QueryNormalizer,
)
from src.common.settings import get_settings


CASES = (
    ("what is clod computin", "what is cloud computing"),
    (
        "How cn I improve Azure reliabiliti?",
        "How can I improve Azure reliability?",
    ),
    ("explan kubernetes archtectur", "explain Kubernetes architecture"),
    ("What is AKS?", "What is AKS?"),
    ("What is cloud computing?", "What is cloud computing?"),
)


class FailingProvider:
    provider_name = "failing-test-provider"
    model_name = "failing-test-model"

    def complete_text(self, **_: object) -> str:
        raise TimeoutError("simulated provider timeout")


def main() -> None:
    settings = get_settings()

    normalizer = QueryNormalizer.from_settings(
        settings
    )

    print("=" * 100)
    print("QUERY NORMALIZER TEST")
    print("=" * 100)

    for question, expected in CASES:
        print()
        print("ORIGINAL   :", question)

        result = normalizer.normalize(
            question
        )

        print("NORMALIZED :", result.normalized_query)
        print("CHANGED    :", result.changed)
        print("PROVIDER   :", result.provider)
        print("MODEL      :", result.model_name)
        assert result.original_query == question
        assert result.normalized_query == expected
        print("-" * 100)

    failing_normalizer = QueryNormalizer(
        provider=FailingProvider(),
        settings=settings.query_processing.normalization,
    )
    original = "what is clod computin"
    fallback = failing_normalizer.normalize(original)
    assert fallback.original_query == original
    assert fallback.normalized_query == original
    assert fallback.changed is False
    print("Fail-safe provider test passed.")


if __name__ == "__main__":
    main()
