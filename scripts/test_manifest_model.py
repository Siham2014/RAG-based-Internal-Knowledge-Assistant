import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.manifest import (
    ManifestEntry,
    ManifestStatus,
)


def main() -> None:
    entry = ManifestEntry.create_new(
        document_id="azure_well_architected:design.md",
        source_id="azure_well_architected",
        source_type="local_folder",
        local_path=(
            "data/raw/markdown/"
            "well-architected/design.md"
        ),
        checksum_sha256="abc123456789",
        file_size_bytes=12045,
    )

    print("=" * 60)
    print("TEST DU MODÈLE MANIFEST")
    print("=" * 60)

    print(f"Document ID : {entry.document_id}")
    print(f"Source ID   : {entry.source_id}")
    print(f"Statut      : {entry.status.value}")
    print(f"Taille      : {entry.file_size_bytes} octets")
    print(f"Créé le     : {entry.first_seen_at}")

    print("\nDictionnaire JSON-compatible :")
    print(entry.to_dict())

    restored_entry = ManifestEntry.from_dict(
        entry.to_dict()
    )

    print("\nEntrée reconstruite :")
    print(f"Document ID : {restored_entry.document_id}")
    print(f"Statut      : {restored_entry.status.value}")

    assert entry.document_id == restored_entry.document_id
    assert entry.status == ManifestStatus.NEW
    assert restored_entry.status == ManifestStatus.NEW

    print("\n Test réussi")


if __name__ == "__main__":
    main()