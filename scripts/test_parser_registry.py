import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.source import SourceConfig
from src.parsing.base import BaseParser, ParserError
from src.parsing.registry import ParserRegistry


@ParserRegistry.register(".txt", ".text")
class TextParser(BaseParser):
    """
    Parser fictif utilisé uniquement pour tester le registre.
    """

    def parse(self, file_path: Path):
        self.validate_file(file_path)
        return None


@ParserRegistry.register("demo")
class DemoParser(BaseParser):
    """
    Deuxième parser fictif.
    """

    def parse(self, file_path: Path):
        self.validate_file(file_path)
        return None


def main() -> None:
    source = SourceConfig(
        id="test_source",
        name="Test Source",
        type="local_folder",
        enabled=True,
        input={},
    )

    print("=" * 60)
    print("TEST DU PARSER REGISTRY")
    print("=" * 60)

    print(
        "Extensions disponibles :",
        ParserRegistry.available_extensions(),
    )

    print(
        "Parsers disponibles :",
        ParserRegistry.available_parsers(),
    )

    txt_file = Path("document.txt")

    assert ParserRegistry.supports(txt_file)
    assert ParserRegistry.supports(
        Path("document.TEXT")
    )
    assert ParserRegistry.supports(
        Path("document.demo")
    )

    parser = ParserRegistry.create(
        file_path=txt_file,
        source=source,
    )

    print(
        "\nParser créé pour document.txt :",
        parser.__class__.__name__,
    )

    assert isinstance(parser, TextParser)

    parser_class = (
        ParserRegistry.get_parser_class(
            Path("notes.text")
        )
    )

    assert parser_class is TextParser

    try:
        ParserRegistry.create(
            file_path=Path("document.pdf"),
            source=source,
        )

    except ParserError as error:
        print("\nErreur attendue :")
        print(error)

    else:
        raise AssertionError(
            "Une erreur devait être levée "
            "pour l'extension .pdf."
        )

    print("\n ParserRegistry fonctionnel.")


if __name__ == "__main__":
    main()