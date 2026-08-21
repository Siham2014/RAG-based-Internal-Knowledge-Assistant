from src.query_processing.query_normalizer import (
    QueryNormalizer,
)


def main() -> None:
    normalizer = QueryNormalizer.from_settings()

    query = "wat is claud compotin"

    result = normalizer.normalize(
        query
    )

    print("=" * 80)
    print("QUERY NORMALIZER")
    print("=" * 80)
    print("Original   :", result.original_query)
    print("Normalized :", result.normalized_query)
    print("Changed    :", result.changed)
    print("Provider   :", result.provider)
    print("Model      :", result.model_name)


if __name__ == "__main__":
    main()
