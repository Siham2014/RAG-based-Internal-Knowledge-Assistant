from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    raw_data: Path
    staging: Path
    processed_documents: Path
    manifests: Path
    rejected: Path
    logs: Path

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        project_root: Path | None = None,
    ) -> "ProjectPaths":
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        paths_config = config.get("paths", {})

        required_paths = [
            "raw_data",
            "staging",
            "processed_documents",
            "manifests",
            "rejected",
            "logs",
        ]

        missing = [
            name
            for name in required_paths
            if name not in paths_config
        ]

        if missing:
            raise ValueError(
                "Chemins absents de ingestion.yaml : "
                + ", ".join(missing)
            )

        return cls(
            project_root=project_root,
            raw_data=project_root / paths_config["raw_data"],
            staging=project_root / paths_config["staging"],
            processed_documents=(
                project_root / paths_config["processed_documents"]
            ),
            manifests=project_root / paths_config["manifests"],
            rejected=project_root / paths_config["rejected"],
            logs=project_root / paths_config["logs"],
        )

    def create_directories(self) -> None:
        """Crée automatiquement tous les dossiers nécessaires."""

        directories = [
            self.raw_data,
            self.staging,
            self.processed_documents,
            self.manifests,
            self.rejected,
            self.logs,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)