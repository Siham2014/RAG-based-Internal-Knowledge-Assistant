from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class ExperimentModelError(ValueError):
    """
    Erreur déclenchée lorsque les informations d'une expérience
    sont invalides.
    """


class ExperimentType(str, Enum):
    """
    Types d'expériences supportés par le projet.
    """

    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    END_TO_END = "end_to_end"


class ExperimentStatus(str, Enum):
    """
    États possibles d'une expérience.
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def utc_now() -> datetime:
    """
    Retourne la date et l'heure actuelles en UTC.
    """

    return datetime.now(timezone.utc)


def normalize_datetime(value: datetime) -> datetime:
    """
    Garantit que la date contient un fuseau horaire.
    """

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


@dataclass(slots=True)
class ExperimentInfo:
    """
    Informations générales décrivant une exécution expérimentale.

    Cette classe ne contient pas les paramètres métier comme :
    - chunk_size ;
    - overlap ;
    - embedding_model ;
    - top_k.

    Ces paramètres restent dans les configurations spécifiques
    de chaque composant.
    """

    experiment_id: str
    name: str
    experiment_type: ExperimentType
    dataset_name: str

    status: ExperimentStatus = ExperimentStatus.CREATED
    description: str | None = None
    version: str = "1.0"

    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_message: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.experiment_id = self.experiment_id.strip()
        self.name = self.name.strip()
        self.dataset_name = self.dataset_name.strip()
        self.version = self.version.strip()

        if isinstance(self.experiment_type, str):
            try:
                self.experiment_type = ExperimentType(
                    self.experiment_type
                )
            except ValueError as exc:
                raise ExperimentModelError(
                    f"Type d'expérience invalide : "
                    f"{self.experiment_type!r}."
                ) from exc

        if isinstance(self.status, str):
            try:
                self.status = ExperimentStatus(self.status)
            except ValueError as exc:
                raise ExperimentModelError(
                    f"Statut d'expérience invalide : "
                    f"{self.status!r}."
                ) from exc

        if not self.experiment_id:
            raise ExperimentModelError(
                "experiment_id ne peut pas être vide."
            )

        if not self.name:
            raise ExperimentModelError(
                "Le nom de l'expérience ne peut pas être vide."
            )

        if not self.dataset_name:
            raise ExperimentModelError(
                "dataset_name ne peut pas être vide."
            )

        if not self.version:
            raise ExperimentModelError(
                "version ne peut pas être vide."
            )

        self.created_at = normalize_datetime(self.created_at)

        if self.started_at is not None:
            self.started_at = normalize_datetime(
                self.started_at
            )

        if self.completed_at is not None:
            self.completed_at = normalize_datetime(
                self.completed_at
            )

        if (
            self.started_at is not None
            and self.started_at < self.created_at
        ):
            raise ExperimentModelError(
                "started_at ne peut pas être antérieur "
                "à created_at."
            )

        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ExperimentModelError(
                "completed_at ne peut pas être antérieur "
                "à started_at."
            )

        self.tags = self._normalize_tags(self.tags)
        self.metadata = dict(self.metadata)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        experiment_type: ExperimentType | str,
        dataset_name: str,
        description: str | None = None,
        version: str = "1.0",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExperimentInfo":
        """
        Crée une nouvelle expérience avec un identifiant unique.
        """

        normalized_name = name.strip().lower()
        normalized_name = "_".join(
            normalized_name.split()
        )

        identifier = uuid4().hex

        experiment_id = (
            f"{experiment_type.value if isinstance(experiment_type, ExperimentType) else experiment_type}"
            f"_{normalized_name}_{identifier}"
        )

        return cls(
            experiment_id=experiment_id,
            name=name,
            experiment_type=experiment_type,
            dataset_name=dataset_name,
            description=description,
            version=version,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )

    def mark_running(
        self,
        *,
        started_at: datetime | None = None,
    ) -> None:
        """
        Passe l'expérience à l'état RUNNING.
        """

        if self.status != ExperimentStatus.CREATED:
            raise ExperimentModelError(
                "Seule une expérience CREATED peut passer "
                "à l'état RUNNING."
            )

        self.status = ExperimentStatus.RUNNING
        self.started_at = normalize_datetime(
            started_at or utc_now()
        )

        self.completed_at = None
        self.error_message = None

    def mark_completed(
        self,
        *,
        completed_at: datetime | None = None,
    ) -> None:
        """
        Passe l'expérience à l'état COMPLETED.
        """

        if self.status != ExperimentStatus.RUNNING:
            raise ExperimentModelError(
                "Seule une expérience RUNNING peut passer "
                "à l'état COMPLETED."
            )

        end_time = normalize_datetime(
            completed_at or utc_now()
        )

        if (
            self.started_at is not None
            and end_time < self.started_at
        ):
            raise ExperimentModelError(
                "La date de fin ne peut pas être antérieure "
                "à la date de début."
            )

        self.status = ExperimentStatus.COMPLETED
        self.completed_at = end_time
        self.error_message = None

    def mark_failed(
        self,
        error_message: str,
        *,
        completed_at: datetime | None = None,
    ) -> None:
        """
        Passe l'expérience à l'état FAILED.
        """

        normalized_error = error_message.strip()

        if not normalized_error:
            raise ExperimentModelError(
                "Le message d'erreur ne peut pas être vide."
            )

        if self.status not in {
            ExperimentStatus.CREATED,
            ExperimentStatus.RUNNING,
        }:
            raise ExperimentModelError(
                "Une expérience terminée ne peut pas passer "
                "à l'état FAILED."
            )

        if self.started_at is None:
            self.started_at = utc_now()

        end_time = normalize_datetime(
            completed_at or utc_now()
        )

        if end_time < self.started_at:
            raise ExperimentModelError(
                "La date d'échec ne peut pas être antérieure "
                "à la date de début."
            )

        self.status = ExperimentStatus.FAILED
        self.completed_at = end_time
        self.error_message = normalized_error

    @property
    def duration_seconds(self) -> float | None:
        """
        Retourne la durée d'exécution en secondes.
        """

        if (
            self.started_at is None
            or self.completed_at is None
        ):
            return None

        duration = (
            self.completed_at - self.started_at
        ).total_seconds()

        return round(duration, 6)

    @property
    def is_finished(self) -> bool:
        return self.status in {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit l'expérience en dictionnaire sérialisable en JSON.
        """

        result = asdict(self)

        result["experiment_type"] = (
            self.experiment_type.value
        )
        result["status"] = self.status.value

        result["created_at"] = (
            self.created_at.isoformat()
        )

        result["started_at"] = (
            self.started_at.isoformat()
            if self.started_at is not None
            else None
        )

        result["completed_at"] = (
            self.completed_at.isoformat()
            if self.completed_at is not None
            else None
        )

        result["duration_seconds"] = (
            self.duration_seconds
        )

        return result

    @staticmethod
    def _normalize_tags(
        tags: list[str],
    ) -> list[str]:
        result: list[str] = []

        for tag in tags:
            normalized = str(tag).strip().lower()

            if normalized and normalized not in result:
                result.append(normalized)

        return result


@dataclass(slots=True)
class ExperimentArtifact:
    """
    Représente un fichier produit par une expérience.

    Exemples :
    - chunks.jsonl ;
    - statistics.json ;
    - config.json ;
    - results.csv ;
    - graphique PNG.
    """

    name: str
    path: str
    artifact_type: str
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.path = self.path.strip()
        self.artifact_type = self.artifact_type.strip().lower()

        if not self.name:
            raise ExperimentModelError(
                "Le nom de l'artifact ne peut pas être vide."
            )

        if not self.path:
            raise ExperimentModelError(
                "Le chemin de l'artifact ne peut pas être vide."
            )

        if not self.artifact_type:
            raise ExperimentModelError(
                "artifact_type ne peut pas être vide."
            )

        self.metadata = dict(self.metadata)

    @property
    def filename(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentResult:
    """
    Résultat générique produit par une expérience.
    """

    experiment: ExperimentInfo

    configuration: dict[str, Any] = field(
        default_factory=dict
    )
    metrics: dict[str, int | float | str | bool | None] = field(
        default_factory=dict
    )
    artifacts: list[ExperimentArtifact] = field(
        default_factory=list
    )
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(
            self.experiment,
            ExperimentInfo,
        ):
            raise ExperimentModelError(
                "experiment doit être une instance "
                "de ExperimentInfo."
            )

        self.configuration = dict(self.configuration)
        self.metrics = dict(self.metrics)
        self.artifacts = list(self.artifacts)
        self.notes = [
            str(note).strip()
            for note in self.notes
            if str(note).strip()
        ]

        if not all(
            isinstance(artifact, ExperimentArtifact)
            for artifact in self.artifacts
        ):
            raise ExperimentModelError(
                "Tous les artifacts doivent être des instances "
                "de ExperimentArtifact."
            )

    def add_metric(
        self,
        name: str,
        value: int | float | str | bool | None,
    ) -> None:
        normalized_name = name.strip()

        if not normalized_name:
            raise ExperimentModelError(
                "Le nom de la métrique ne peut pas être vide."
            )

        self.metrics[normalized_name] = value

    def add_artifact(
        self,
        artifact: ExperimentArtifact,
    ) -> None:
        if not isinstance(
            artifact,
            ExperimentArtifact,
        ):
            raise ExperimentModelError(
                "artifact doit être une instance "
                "de ExperimentArtifact."
            )

        self.artifacts.append(artifact)

    def add_note(
        self,
        note: str,
    ) -> None:
        normalized_note = note.strip()

        if not normalized_note:
            raise ExperimentModelError(
                "La note ne peut pas être vide."
            )

        self.notes.append(normalized_note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "configuration": dict(self.configuration),
            "metrics": dict(self.metrics),
            "artifacts": [
                artifact.to_dict()
                for artifact in self.artifacts
            ],
            "notes": list(self.notes),
        }