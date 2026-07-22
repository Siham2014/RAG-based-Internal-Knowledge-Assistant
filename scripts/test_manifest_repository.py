import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.manifest.repository import (
    ManifestRepository,
)
from src.models.manifest import ManifestEntry


def main():

    repository = ManifestRepository(
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "ingestion_manifest.json"
    )

    entry = ManifestEntry.create_new(
        document_id="demo",
        source_id="local",
        source_type="local_folder",
        local_path="demo.md",
        checksum_sha256="123456",
    )

    repository.upsert(entry)

    loaded = repository.load()

    print("=" * 60)
    print(f"Nombre d'entrées : {len(loaded)}")
    print(list(loaded.keys()))

    print("\nDocument chargé :")
    print(repository.get("demo"))

    print("\nTest réussi.")
    

if __name__ == "__main__":
    main()