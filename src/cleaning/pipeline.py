from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from src.cleaning.base import (
    BaseCleaner,
    CleaningError,
)
from src.cleaning.registry import CleanerRegistry
from src.models.document import ParsedDocument


@dataclass
class CleanerExecutionReport:
    """
    Rapport d'exécution d'un cleaner individuel.
    """

    cleaner_name: str
    enabled: bool
    success: bool
    duration_ms: float
    error_message: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit le rapport en dictionnaire.
        """

        return {
            "cleaner_name": self.cleaner_name,
            "enabled": self.enabled,
            "success": self.success,
            "duration_ms": round(
                self.duration_ms,
                3,
            ),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class CleaningPipelineReport:
    """
    Rapport global du pipeline de nettoyage.
    """

    pipeline_name: str
    success: bool
    started_at: str
    completed_at: str
    duration_ms: float
    cleaners: list[CleanerExecutionReport]
    input_section_count: int
    output_section_count: int
    input_character_count: int
    output_character_count: int
    error_message: str | None = None

    @property
    def removed_section_count(self) -> int:
        """
        Nombre total de sections supprimées.
        """

        return max(
            0,
            self.input_section_count
            - self.output_section_count,
        )

    @property
    def removed_character_count(self) -> int:
        """
        Nombre approximatif de caractères supprimés.
        """

        return max(
            0,
            self.input_character_count
            - self.output_character_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit le rapport complet en dictionnaire.
        """

        return {
            "pipeline_name": self.pipeline_name,
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(
                self.duration_ms,
                3,
            ),
            "input_section_count": (
                self.input_section_count
            ),
            "output_section_count": (
                self.output_section_count
            ),
            "removed_section_count": (
                self.removed_section_count
            ),
            "input_character_count": (
                self.input_character_count
            ),
            "output_character_count": (
                self.output_character_count
            ),
            "removed_character_count": (
                self.removed_character_count
            ),
            "error_message": self.error_message,
            "cleaners": [
                cleaner_report.to_dict()
                for cleaner_report in self.cleaners
            ],
        }


class CleaningPipelineError(Exception):
    """
    Erreur levée par le pipeline de nettoyage.
    """


class CleaningPipeline:
    """
    Exécute plusieurs cleaners sur un ParsedDocument.

    Le pipeline :

    - respecte l'ordre des cleaners ;
    - conserve l'objet original ;
    - mesure le temps d'exécution ;
    - produit un rapport détaillé ;
    - peut arrêter ou continuer en cas d'erreur.
    """

    DEFAULT_CLEANERS = [
        {
            "name": "unicode",
            "enabled": True,
            "config": {},
        },
        {
            "name": "whitespace",
            "enabled": True,
            "config": {},
        },
        {
            "name": "boilerplate",
            "enabled": True,
            "config": {},
        },
        {
            "name": "duplicate",
            "enabled": True,
            "config": {
                "global_scope": False,
                "remove_content_empty_sections": True,
            },
        },
    ]

    def __init__(
        self,
        cleaners: list[
            BaseCleaner | str | dict[str, Any]
        ] | None = None,
        name: str = "default_cleaning_pipeline",
        stop_on_error: bool = True,
    ) -> None:
        self.name = name
        self.stop_on_error = stop_on_error

        cleaner_definitions = (
            cleaners
            if cleaners is not None
            else deepcopy(self.DEFAULT_CLEANERS)
        )

        self.cleaners = self._build_cleaners(
            cleaner_definitions
        )

    def run(
        self,
        document: ParsedDocument,
    ) -> tuple[
        ParsedDocument,
        CleaningPipelineReport,
    ]:
        """
        Exécute les cleaners sur une copie du document.

        Retourne :

        - le document nettoyé ;
        - le rapport d'exécution.
        """

        self._validate_document(document)

        pipeline_start = perf_counter()

        started_at = datetime.now(
            timezone.utc
        ).isoformat()

        input_section_count = len(
            document.sections
        )

        input_character_count = (
            self._count_document_characters(
                document
            )
        )

        working_document = deepcopy(document)

        cleaner_reports: list[
            CleanerExecutionReport
        ] = []

        pipeline_success = True
        pipeline_error_message = None

        for cleaner in self.cleaners:
            cleaner_start = perf_counter()

            try:
                working_document = cleaner.run(
                    working_document
                )

                duration_ms = (
                    perf_counter()
                    - cleaner_start
                ) * 1000

                cleaner_reports.append(
                    CleanerExecutionReport(
                        cleaner_name=(
                            cleaner.cleaner_name
                        ),
                        enabled=cleaner.enabled,
                        success=True,
                        duration_ms=duration_ms,
                        metadata=(
                            self._extract_cleaner_metadata(
                                working_document,
                                cleaner,
                            )
                        ),
                    )
                )

            except Exception as error:
                duration_ms = (
                    perf_counter()
                    - cleaner_start
                ) * 1000

                error_message = str(error)

                cleaner_reports.append(
                    CleanerExecutionReport(
                        cleaner_name=(
                            cleaner.cleaner_name
                        ),
                        enabled=cleaner.enabled,
                        success=False,
                        duration_ms=duration_ms,
                        error_message=error_message,
                    )
                )

                pipeline_success = False
                pipeline_error_message = (
                    f"Cleaner "
                    f"'{cleaner.cleaner_name}' "
                    f"failed: {error_message}"
                )

                if self.stop_on_error:
                    break

        completed_at = datetime.now(
            timezone.utc
        ).isoformat()

        pipeline_duration_ms = (
            perf_counter()
            - pipeline_start
        ) * 1000

        output_section_count = len(
            working_document.sections
        )

        output_character_count = (
            self._count_document_characters(
                working_document
            )
        )

        report = CleaningPipelineReport(
            pipeline_name=self.name,
            success=pipeline_success,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=pipeline_duration_ms,
            cleaners=cleaner_reports,
            input_section_count=(
                input_section_count
            ),
            output_section_count=(
                output_section_count
            ),
            input_character_count=(
                input_character_count
            ),
            output_character_count=(
                output_character_count
            ),
            error_message=(
                pipeline_error_message
            ),
        )

        if working_document.metadata is None:
            working_document.metadata = {}

        working_document.metadata[
            "cleaning_pipeline"
        ] = report.to_dict()

        working_document.metadata[
            "cleaning_pipeline_applied"
        ] = True

        working_document.metadata[
            "cleaning_pipeline_name"
        ] = self.name

        if (
            not pipeline_success
            and self.stop_on_error
        ):
            raise CleaningPipelineError(
                pipeline_error_message
                or "Cleaning pipeline failed."
            )

        return working_document, report

    def _build_cleaners(
        self,
        definitions: list[
            BaseCleaner | str | dict[str, Any]
        ],
    ) -> list[BaseCleaner]:
        """
        Transforme les définitions en instances de cleaners.
        """

        cleaners: list[BaseCleaner] = []

        for definition in definitions:
            if isinstance(
                definition,
                BaseCleaner,
            ):
                cleaners.append(definition)
                continue

            if isinstance(definition, str):
                cleaner = CleanerRegistry.create(
                    definition
                )

                cleaners.append(cleaner)
                continue

            if isinstance(definition, dict):
                cleaner_name = definition.get(
                    "name"
                )

                if not cleaner_name:
                    raise CleaningPipelineError(
                        "Chaque définition de cleaner "
                        "doit contenir une clé 'name'."
                    )

                enabled = definition.get(
                    "enabled",
                    True,
                )

                config = definition.get(
                    "config",
                    {},
                )

                cleaner = CleanerRegistry.create(
                    cleaner_name,
                    enabled=enabled,
                    config=config,
                )

                cleaners.append(cleaner)
                continue

            raise CleaningPipelineError(
                "Définition de cleaner invalide : "
                f"{definition!r}"
            )

        if not cleaners:
            raise CleaningPipelineError(
                "Le pipeline doit contenir au moins "
                "un cleaner."
            )

        return cleaners

    @staticmethod
    def _validate_document(
        document: ParsedDocument,
    ) -> None:
        """
        Vérifie le document d'entrée.
        """

        if not isinstance(
            document,
            ParsedDocument,
        ):
            raise CleaningPipelineError(
                "CleaningPipeline attend une "
                "instance de ParsedDocument."
            )

        if not document.document_id:
            raise CleaningPipelineError(
                "Le document doit posséder "
                "un document_id."
            )

        if document.sections is None:
            raise CleaningPipelineError(
                "Le champ sections ne peut pas "
                "être None."
            )

    @staticmethod
    def _count_document_characters(
        document: ParsedDocument,
    ) -> int:
        """
        Compte approximativement tous les caractères
        textuels du document.
        """

        total = 0

        top_level_values = [
            document.title,
            document.author,
            document.source_url,
            document.language,
            document.organization,
            document.local_path,
        ]

        for value in top_level_values:
            if value:
                total += len(str(value))

        for section in document.sections:
            if section.heading:
                total += len(section.heading)

            total += sum(
                len(value)
                for value in section.heading_path
                if value
            )

            total += sum(
                len(paragraph)
                for paragraph in section.paragraphs
                if paragraph
            )

            for items in section.lists:
                total += sum(
                    len(item)
                    for item in items
                    if item
                )

            for table in section.tables:
                total += sum(
                    len(header)
                    for header in table.headers
                    if header
                )

                for row in table.rows:
                    total += sum(
                        len(cell)
                        for cell in row
                        if cell
                    )

            for code_block in (
                section.code_blocks
            ):
                if code_block.language:
                    total += len(
                        code_block.language
                    )

                if code_block.content:
                    total += len(
                        code_block.content
                    )

            for link in section.links:
                if link.text:
                    total += len(link.text)

                if link.url:
                    total += len(link.url)

        return total

    @staticmethod
    def _extract_cleaner_metadata(
        document: ParsedDocument,
        cleaner: BaseCleaner,
    ) -> dict[str, Any]:
        """
        Extrait les métadonnées liées à un cleaner.
        """

        metadata = document.metadata or {}

        result: dict[str, Any] = {}

        cleaner_metadata_keys = {
            "unicode_cleaner": [],
            "whitespace_cleaner": [],
            "boilerplate_cleaner": [
                "boilerplate_removed",
            ],
            "duplicate_cleaner": [
                "duplicates_removed",
                "duplicate_cleaner_config",
            ],
        }

        keys = cleaner_metadata_keys.get(
            cleaner.cleaner_name,
            [],
        )

        for key in keys:
            if key in metadata:
                result[key] = deepcopy(
                    metadata[key]
                )

        return result

    def describe(self) -> dict[str, Any]:
        """
        Retourne la configuration du pipeline.
        """

        return {
            "name": self.name,
            "stop_on_error": self.stop_on_error,
            "cleaner_count": len(
                self.cleaners
            ),
            "cleaners": [
                {
                    "cleaner_name": (
                        cleaner.cleaner_name
                    ),
                    "enabled": cleaner.enabled,
                    "config": deepcopy(
                        cleaner.config
                    ),
                }
                for cleaner in self.cleaners
            ],
        }