import hashlib
from pathlib import Path

from src.ingestion.manifest.repository import (
    ManifestRepository,
)
from src.models.manifest import (
    ManifestEntry,
    ManifestStatus,
)


class ManifestService:
    """
    Contient la logique métier du Manifest.

    Il permet de :
    - calculer le SHA-256 d'un fichier ;
    - déterminer si un fichier est nouveau ;
    - détecter une modification ;
    - détecter un fichier inchangé ;
    - enregistrer le résultat dans le repository.
    """

    def __init__(
        self,
        repository: ManifestRepository,
    ) -> None:
        self.repository = repository

    @staticmethod
    def calculate_sha256(
        file_path: Path,
        block_size: int = 65536,
    ) -> str:
        """
        Calcule le SHA-256 d'un fichier.

        Le fichier est lu par blocs afin d'éviter de charger
        un gros PDF ou un gros fichier HTML entièrement en mémoire.
        """

        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Le chemin n'est pas un fichier : {file_path}"
            )

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            while True:
                block = file.read(block_size)

                if not block:
                    break

                sha256.update(block)

        return sha256.hexdigest()

    def evaluate_file(
        self,
        document_id: str,
        source_id: str,
        source_type: str,
        file_path: Path,
        source_url: str | None = None,
        remote_version: str | None = None,
    ) -> ManifestEntry:
        """
        Compare un fichier avec son entrée existante.

        Retourne une entrée avec le statut :
        - NEW ;
        - MODIFIED ;
        - UNCHANGED.
        """

        resolved_path = file_path.resolve()

        checksum = self.calculate_sha256(
            resolved_path
        )

        file_size = resolved_path.stat().st_size

        existing_entry = self.repository.get(
            document_id
        )

        if existing_entry is None:
            entry = ManifestEntry.create_new(
                document_id=document_id,
                source_id=source_id,
                source_type=source_type,
                local_path=str(resolved_path),
                checksum_sha256=checksum,
                source_url=source_url,
                remote_version=remote_version,
                file_size_bytes=file_size,
            )

            self.repository.upsert(entry)

            return entry

        now = ManifestEntry.utc_now()

        existing_entry.source_id = source_id
        existing_entry.source_type = source_type
        existing_entry.local_path = str(resolved_path)
        existing_entry.source_url = source_url
        existing_entry.remote_version = remote_version
        existing_entry.file_size_bytes = file_size
        existing_entry.last_seen_at = now
        existing_entry.error_message = None

        if existing_entry.checksum_sha256 == checksum:
            existing_entry.status = (
                ManifestStatus.UNCHANGED
            )
        else:
            existing_entry.status = (
                ManifestStatus.MODIFIED
            )
            existing_entry.checksum_sha256 = checksum
            existing_entry.ingested_at = now

        self.repository.upsert(
            existing_entry
        )

        return existing_entry

    def mark_missing(
        self,
        document_id: str,
    ) -> ManifestEntry:
        """
        Marque un document comme manquant.
        """

        entry = self.repository.get(document_id)

        if entry is None:
            raise KeyError(
                "Impossible de marquer comme manquant "
                f"un document absent : {document_id}"
            )

        entry.status = ManifestStatus.MISSING
        entry.last_seen_at = ManifestEntry.utc_now()

        self.repository.upsert(entry)

        return entry

    def mark_failed(
        self,
        document_id: str,
        error_message: str,
    ) -> ManifestEntry:
        """
        Marque une entrée existante comme échouée.
        """

        entry = self.repository.get(document_id)

        if entry is None:
            raise KeyError(
                "Impossible de marquer comme échoué "
                f"un document absent : {document_id}"
            )

        entry.status = ManifestStatus.FAILED
        entry.error_message = error_message
        entry.last_seen_at = ManifestEntry.utc_now()

        self.repository.upsert(entry)

        return entry