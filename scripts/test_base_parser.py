import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.source import SourceConfig
from src.parsing.base import BaseParser


class DummyParser(BaseParser):
    supported_extensions = {".txt"}

    def parse(self, file_path: Path):
        self.validate_file(file_path)
        return None


def main() -> None:
    source = SourceConfig(
        id="dummy",
        name="Dummy",
        type="local_folder",
        enabled=True,
        input={},
    )

    parser = DummyParser(source)

    document_id = parser.build_document_id(
        Path("example.txt")
    )

    print(document_id)

    assert document_id == "dummy:example"

    print(" Test BaseParser réussi")


if __name__ == "__main__":
    main()