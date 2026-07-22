from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ManifestStatus(str, Enum):
    """
    États possibles d'un document dans le manifest.
    """

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    MISSING = "missing"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ManifestEntry:
    """
    Représente un document suivi par le pipeline d'ingestion.

    Une entrée du manifest permet de savoir si un document est :
    - nouveau ;
    - modifié ;
    - inchangé ;
    - manquant ;
    - rejeté ;
    - en erreur.
    """

    document_id: str
    source_id: str
    source_type: str
    local_path: str
    checksum_sha256: str
    status: ManifestStatus

    source_url: str | None = None
    remote_version: str | None = None
    file_size_bytes: int = 0

    first_seen_at: str = ""
    last_seen_at: str = ""
    ingested_at: str = ""

    error_message: str | None = None

    @staticmethod
    def utc_now() -> str:
        """
        Retourne la date actuelle au format ISO 8601 en UTC.
        """

        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def create_new(
        cls,
        document_id: str,
        source_id: str,
        source_type: str,
        local_path: str,
        checksum_sha256: str,
        source_url: str | None = None,
        remote_version: str | None = None,
        file_size_bytes: int = 0,
    ) -> "ManifestEntry":
        """
        Crée une nouvelle entrée de manifest.
        """

        now = cls.utc_now()

        return cls(
            document_id=document_id,
            source_id=source_id,
            source_type=source_type,
            local_path=local_path,
            checksum_sha256=checksum_sha256,
            status=ManifestStatus.NEW,
            source_url=source_url,
            remote_version=remote_version,
            file_size_bytes=file_size_bytes,
            first_seen_at=now,
            last_seen_at=now,
            ingested_at=now,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit l'entrée en dictionnaire JSON-compatible.
        """

        data = asdict(self)
        data["status"] = self.status.value

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ManifestEntry":
        """
        Reconstruit une entrée depuis un dictionnaire JSON.
        """

        return cls(
            document_id=str(data["document_id"]),
            source_id=str(data["source_id"]),
            source_type=str(data["source_type"]),
            local_path=str(data["local_path"]),
            checksum_sha256=str(
                data.get("checksum_sha256", "")
            ),
            status=ManifestStatus(
                data.get(
                    "status",
                    ManifestStatus.NEW.value,
                )
            ),
            source_url=data.get("source_url"),
            remote_version=data.get("remote_version"),
            file_size_bytes=int(
                data.get("file_size_bytes", 0)
            ),
            first_seen_at=str(
                data.get("first_seen_at", "")
            ),
            last_seen_at=str(
                data.get("last_seen_at", "")
            ),
            ingested_at=str(
                data.get("ingested_at", "")
            ),
            error_message=data.get("error_message"),
        )