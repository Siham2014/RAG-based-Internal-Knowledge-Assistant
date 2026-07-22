import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.manifest.repository import (
    ManifestRepository,
)
from src.ingestion.manifest.service import (
    ManifestService,
)
from src.models.manifest import ManifestStatus


def main() -> None:
    test_directory = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "manifest_test"
    )

    test_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_file = test_directory / "document_test.md"

    manifest_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "test_ingestion_manifest.json"
    )

    if manifest_path.exists():
        manifest_path.unlink()

    repository = ManifestRepository(
        manifest_path=manifest_path
    )

    service = ManifestService(
        repository=repository
    )

    print("=" * 60)
    print("TEST DU MANIFEST SERVICE")
    print("=" * 60)

    # Premier contenu : le fichier doit être NEW.
    test_file.write_text(
        "# Azure\n\nPremier contenu.",
        encoding="utf-8",
    )

    first_entry = service.evaluate_file(
        document_id="test:document_test.md",
        source_id="test_source",
        source_type="local_folder",
        file_path=test_file,
    )

    print(
        "Premier passage :",
        first_entry.status.value,
    )

    assert first_entry.status == ManifestStatus.NEW

    # Deuxième passage sans modification : UNCHANGED.
    second_entry = service.evaluate_file(
        document_id="test:document_test.md",
        source_id="test_source",
        source_type="local_folder",
        file_path=test_file,
    )

    print(
        "Deuxième passage :",
        second_entry.status.value,
    )

    assert (
        second_entry.status
        == ManifestStatus.UNCHANGED
    )

    # Modification réelle du contenu : MODIFIED.
    test_file.write_text(
        "# Azure\n\nContenu modifié.",
        encoding="utf-8",
    )

    third_entry = service.evaluate_file(
        document_id="test:document_test.md",
        source_id="test_source",
        source_type="local_folder",
        file_path=test_file,
    )

    print(
        "Après modification :",
        third_entry.status.value,
    )

    assert (
        third_entry.status
        == ManifestStatus.MODIFIED
    )

    # Nouveau passage sans modification : UNCHANGED.
    fourth_entry = service.evaluate_file(
        document_id="test:document_test.md",
        source_id="test_source",
        source_type="local_folder",
        file_path=test_file,
    )

    print(
        "Passage suivant :",
        fourth_entry.status.value,
    )

    assert (
        fourth_entry.status
        == ManifestStatus.UNCHANGED
    )

    missing_entry = service.mark_missing(
        "test:document_test.md"
    )

    print(
        "Après mark_missing :",
        missing_entry.status.value,
    )

    assert (
        missing_entry.status
        == ManifestStatus.MISSING
    )

    print("\nSHA-256 :")
    print(fourth_entry.checksum_sha256)

    print("\nManifest créé dans :")
    print(manifest_path)

    print("\n Tous les tests sont réussis.")


if __name__ == "__main__":
    main()