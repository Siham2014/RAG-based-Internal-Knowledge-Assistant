from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable

from src.cleaning.pipeline import (
    CleaningPipeline,
    CleaningPipelineReport,
)
from src.models.document import ParsedDocument


@dataclass
class DocumentCleaningResult:
    """
    Résultat du nettoyage d'un document individuel.
    """

    document_id: str
    success: bool
    cleaned_document: ParsedDocument | None
    report: CleaningPipelineReport | None
    duration_ms: float
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "success": self.success,
            "duration_ms": round(
                self.duration_ms,
                3,
            ),
            "error_message": self.error_message,
            "report": (
                self.report.to_dict()
                if self.report is not None
                else None
            ),
        }


@dataclass
class CleaningServiceReport:
    """
    Rapport global du nettoyage de plusieurs documents.
    """

    service_name: str
    success: bool
    started_at: str
    completed_at: str
    duration_ms: float
    total_documents: int
    successful_documents: int
    failed_documents: int
    input_sections: int
    output_sections: int
    results: list[DocumentCleaningResult] = field(
        default_factory=list
    )

    @property
    def removed_sections(self) -> int:
        return max(
            0,
            self.input_sections
            - self.output_sections,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_name": self.service_name,
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(
                self.duration_ms,
                3,
            ),
            "total_documents": self.total_documents,
            "successful_documents": (
                self.successful_documents
            ),
            "failed_documents": self.failed_documents,
            "input_sections": self.input_sections,
            "output_sections": self.output_sections,
            "removed_sections": self.removed_sections,
            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


class CleaningServiceError(Exception):
    """
    Erreur générale du service de nettoyage.
    """


class CleaningService:
    """
    Service chargé de nettoyer plusieurs documents.

    Il utilise un CleaningPipeline pour chaque document.

    Le service peut :

    - nettoyer un document ;
    - nettoyer plusieurs documents ;
    - ignorer ou arrêter le traitement en cas d'erreur ;
    - produire un rapport global ;
    - conserver l'ordre des documents.
    """

    def __init__(
        self,
        pipeline: CleaningPipeline | None = None,
        name: str = "default_cleaning_service",
        stop_on_error: bool = False,
    ) -> None:
        self.pipeline = (
            pipeline
            if pipeline is not None
            else CleaningPipeline()
        )

        self.name = name
        self.stop_on_error = stop_on_error

    def clean_document(
        self,
        document: ParsedDocument,
    ) -> DocumentCleaningResult:
        """
        Nettoie un seul document.
        """

        start = perf_counter()

        document_id = getattr(
            document,
            "document_id",
            "unknown",
        )

        try:
            cleaned_document, report = (
                self.pipeline.run(document)
            )

            duration_ms = (
                perf_counter() - start
            ) * 1000

            return DocumentCleaningResult(
                document_id=document_id,
                success=True,
                cleaned_document=cleaned_document,
                report=report,
                duration_ms=duration_ms,
            )

        except Exception as error:
            duration_ms = (
                perf_counter() - start
            ) * 1000

            return DocumentCleaningResult(
                document_id=document_id,
                success=False,
                cleaned_document=None,
                report=None,
                duration_ms=duration_ms,
                error_message=str(error),
            )

    def clean_documents(
        self,
        documents: Iterable[ParsedDocument],
    ) -> tuple[
        list[ParsedDocument],
        CleaningServiceReport,
    ]:
        """
        Nettoie plusieurs documents.

        Retourne :

        - la liste des documents nettoyés avec succès ;
        - le rapport global du service.
        """

        service_start = perf_counter()

        started_at = datetime.now(
            timezone.utc
        ).isoformat()

        document_list = list(documents)

        self._validate_documents(
            document_list
        )

        results: list[
            DocumentCleaningResult
        ] = []

        cleaned_documents: list[
            ParsedDocument
        ] = []

        input_sections = sum(
            len(document.sections)
            for document in document_list
        )

        output_sections = 0

        for document in document_list:
            result = self.clean_document(
                document
            )

            results.append(result)

            if result.success:
                if result.cleaned_document is not None:
                    cleaned_documents.append(
                        result.cleaned_document
                    )

                    output_sections += len(
                        result.cleaned_document.sections
                    )

            elif self.stop_on_error:
                raise CleaningServiceError(
                    "Le nettoyage du document "
                    f"'{result.document_id}' a échoué : "
                    f"{result.error_message}"
                )

        completed_at = datetime.now(
            timezone.utc
        ).isoformat()

        duration_ms = (
            perf_counter() - service_start
        ) * 1000

        successful_documents = sum(
            1
            for result in results
            if result.success
        )

        failed_documents = (
            len(results)
            - successful_documents
        )

        report = CleaningServiceReport(
            service_name=self.name,
            success=failed_documents == 0,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            total_documents=len(
                document_list
            ),
            successful_documents=(
                successful_documents
            ),
            failed_documents=(
                failed_documents
            ),
            input_sections=input_sections,
            output_sections=output_sections,
            results=results,
        )

        for cleaned_document in cleaned_documents:
            if cleaned_document.metadata is None:
                cleaned_document.metadata = {}

            cleaned_document.metadata[
                "cleaning_service_applied"
            ] = True

            cleaned_document.metadata[
                "cleaning_service_name"
            ] = self.name

        return cleaned_documents, report

    @staticmethod
    def _validate_documents(
        documents: list[ParsedDocument],
    ) -> None:
        """
        Vérifie la liste des documents.
        """

        if not isinstance(
            documents,
            list,
        ):
            raise CleaningServiceError(
                "Les documents doivent être "
                "fournis sous forme de liste."
            )

        if not documents:
            raise CleaningServiceError(
                "La liste des documents est vide."
            )

        for index, document in enumerate(
            documents
        ):
            if not isinstance(
                document,
                ParsedDocument,
            ):
                raise CleaningServiceError(
                    "L'élément à l'index "
                    f"{index} n'est pas un "
                    "ParsedDocument."
                )

    def describe(self) -> dict[str, Any]:
        """
        Retourne la configuration du service.
        """

        return {
            "name": self.name,
            "stop_on_error": self.stop_on_error,
            "pipeline": deepcopy(
                self.pipeline.describe()
            ),
        }