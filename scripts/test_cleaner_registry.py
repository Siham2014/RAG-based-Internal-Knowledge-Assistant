import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.cleaning import (
    BaseCleaner,
    CleanerRegistry,
    CleaningError,
)
from src.models.document import (
    ParsedDocument,
)


@CleanerRegistry.register(
    "test",
    "test_cleaner",
)
class TestCleaner(BaseCleaner):
    cleaner_name = "test_cleaner"

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        document.title = (
            f"{document.title} - registry"
        )

        return document


@CleanerRegistry.register(
    "disabled_demo"
)
class DisabledDemoCleaner(BaseCleaner):
    cleaner_name = "disabled_demo"

    def clean(
        self,
        document: ParsedDocument,
    ) -> ParsedDocument:
        document.title = "modified"

        return document


def main() -> None:
    print("=" * 70)
    print("TEST DU CLEANER REGISTRY")
    print("=" * 70)

    print(
        "\nCleaners disponibles :",
        CleanerRegistry.available_cleaners(),
    )

    assert CleanerRegistry.supports(
        "test"
    )

    assert CleanerRegistry.supports(
        "TEST"
    )

    assert CleanerRegistry.supports(
        "test-cleaner"
    )

    assert not CleanerRegistry.supports(
        "unknown"
    )

    cleaner_class = (
        CleanerRegistry.get_cleaner_class(
            "test"
        )
    )

    print(
        "\nClasse trouvée :",
        cleaner_class.__name__,
    )

    assert cleaner_class is TestCleaner

    cleaner = CleanerRegistry.create(
        "test",
        enabled=True,
        config={
            "example": True,
        },
    )

    print(
        "\nInstance créée :",
        cleaner.describe(),
    )

    assert isinstance(
        cleaner,
        TestCleaner,
    )

    assert cleaner.enabled is True

    assert cleaner.config == {
        "example": True,
    }

    disabled_cleaner = (
        CleanerRegistry.create(
            "disabled-demo",
            enabled=False,
        )
    )

    assert isinstance(
        disabled_cleaner,
        DisabledDemoCleaner,
    )

    assert disabled_cleaner.enabled is False

    normalized_name = (
        CleanerRegistry.normalize_name(
            " Unicode Cleaner "
        )
    )

    print(
        "\nNom normalisé :",
        normalized_name,
    )

    assert normalized_name == (
        "unicode_cleaner"
    )

    try:
        CleanerRegistry.create(
            "not_found"
        )

    except CleaningError as error:
        print(
            "\nErreur attendue :",
            error,
        )

    else:
        raise AssertionError(
            "Une CleaningError était attendue."
        )

    print(
        "\n CleanerRegistry fonctionnel."
    )


if __name__ == "__main__":
    main()