import json
from pathlib import Path

from src.models.manifest import ManifestEntry


class ManifestRepository:
    """
    Gère la lecture et l'écriture du fichier
    ingestion_manifest.json.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def load(self) -> dict[str, ManifestEntry]:
        """
        Charge le manifest depuis le disque.
        """

        if not self.manifest_path.exists():
            return {}

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        entries: dict[str, ManifestEntry] = {}

        for item in data:
            entry = ManifestEntry.from_dict(item)
            entries[entry.document_id] = entry

        return entries

    def save(
        self,
        entries: dict[str, ManifestEntry],
    ) -> None:
        """
        Sauvegarde le manifest.
        """

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = [
            entry.to_dict()
            for entry in entries.values()
        ]

        serialized.sort(
            key=lambda item: item["document_id"]
        )

        with open(
            self.manifest_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                serialized,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Vérifie si un document existe.
        """

        return document_id in self.load()

    def get(
        self,
        document_id: str,
    ) -> ManifestEntry | None:
        """
        Retourne une entrée.
        """

        return self.load().get(document_id)

    def upsert(
        self,
        entry: ManifestEntry,
    ) -> None:
        """
        Ajoute ou met à jour une entrée.
        """

        entries = self.load()

        entries[entry.document_id] = entry

        self.save(entries)

    def delete(
        self,
        document_id: str,
    ) -> None:
        """
        Supprime une entrée.
        """

        entries = self.load()

        if document_id in entries:
            del entries[document_id]

        self.save(entries)