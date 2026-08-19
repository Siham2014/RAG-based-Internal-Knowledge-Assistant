from src.query_processing.spell_corrector import (
    QuerySpellCorrector,
)


TESTS = [
    "what is clod computin",
    "what is claud compoting",
    "How can I get better Azure reliabiliti?",
    "What is Azure Databrikc?",
    "Explain Kuburnit on Azure",
    "What is AKc?",
    "Explain cloud scalabiliti",
    "How to improve reliability",
]


def main() -> None:

    corrector = QuerySpellCorrector()

    print("=" * 80)
    print("QUERY SPELL CORRECTOR TEST")
    print("=" * 80)

    for query in TESTS:

        corrected = corrector.normalize(
            query
        )

        print()
        print("Original  :", query)
        print("Corrected :", corrected)


if __name__ == "__main__":
    main()