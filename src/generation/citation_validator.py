from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.generation.models import (
    GenerationContext,
    GenerationResponse,
)


CITATION_PATTERN = re.compile(
    r"\[([^\[\]:]+):([^\[\]]+)\]"
)


@dataclass(frozen=True)
class CitationValidationResult:
    """
    Résultat de la validation des citations d'une réponse.
    """

    valid: bool

    detected_citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    missing_citations: tuple[str, ...]

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "detected_citations": list(
                self.detected_citations
            ),
            "valid_citations": list(
                self.valid_citations
            ),
            "invalid_citations": list(
                self.invalid_citations
            ),
            "missing_citations": list(
                self.missing_citations
            ),
            "reason": self.reason,
        }


class CitationValidator:
    """
    Vérifie que les citations produites par le générateur
    correspondent uniquement aux contextes autorisés.

    Format attendu :

        [source:chunk_id]
    """

    def __init__(
        self,
        require_at_least_one_citation: bool = True,
    ) -> None:
        if not isinstance(
            require_at_least_one_citation,
            bool,
        ):
            raise TypeError(
                "require_at_least_one_citation doit "
                "être un booléen."
            )

        self.require_at_least_one_citation = (
            require_at_least_one_citation
        )

    @staticmethod
    def extract_citations(
        answer: str,
    ) -> tuple[str, ...]:
        """
        Extrait toutes les citations présentes dans une réponse.
        """

        normalized_answer = str(
            answer or ""
        )

        citations: list[str] = []

        for match in CITATION_PATTERN.finditer(
            normalized_answer
        ):
            source = match.group(1).strip()
            chunk_id = match.group(2).strip()

            citation = (
                f"[{source}:{chunk_id}]"
            )

            if citation not in citations:
                citations.append(
                    citation
                )

        return tuple(
            citations
        )

    @staticmethod
    def authorized_citations(
        contexts: tuple[
            GenerationContext,
            ...,
        ],
    ) -> tuple[str, ...]:
        """
        Retourne les citations autorisées à partir
        des contextes transmis au LLM.
        """

        return tuple(
            context.citation_id
            for context in contexts
        )

    def validate_answer(
        self,
        answer: str,
        contexts: tuple[
            GenerationContext,
            ...,
        ],
    ) -> CitationValidationResult:
        """
        Valide les citations d'un texte généré.
        """

        if not str(answer).strip():
            return CitationValidationResult(
                valid=False,
                detected_citations=(),
                valid_citations=(),
                invalid_citations=(),
                missing_citations=(),
                reason="La réponse générée est vide.",
            )

        if not contexts:
            return CitationValidationResult(
                valid=False,
                detected_citations=(),
                valid_citations=(),
                invalid_citations=(),
                missing_citations=(),
                reason=(
                    "Aucun contexte n'a été fourni "
                    "pour valider les citations."
                ),
            )

        detected = self.extract_citations(
            answer
        )

        authorized = set(
            self.authorized_citations(
                contexts
            )
        )

        valid_citations = tuple(
            citation
            for citation in detected
            if citation in authorized
        )

        invalid_citations = tuple(
            citation
            for citation in detected
            if citation not in authorized
        )

        if (
            self.require_at_least_one_citation
            and not detected
        ):
            return CitationValidationResult(
                valid=False,
                detected_citations=detected,
                valid_citations=valid_citations,
                invalid_citations=invalid_citations,
                missing_citations=tuple(
                    authorized
                ),
                reason=(
                    "La réponse ne contient aucune citation."
                ),
            )

        if invalid_citations:
            return CitationValidationResult(
                valid=False,
                detected_citations=detected,
                valid_citations=valid_citations,
                invalid_citations=invalid_citations,
                missing_citations=(),
                reason=(
                    "La réponse contient une ou plusieurs "
                    "citations non autorisées."
                ),
            )

        return CitationValidationResult(
            valid=True,
            detected_citations=detected,
            valid_citations=valid_citations,
            invalid_citations=(),
            missing_citations=(),
            reason=(
                "Toutes les citations détectées "
                "sont valides."
            ),
        )

    def validate_response(
        self,
        response: GenerationResponse,
        contexts: tuple[
            GenerationContext,
            ...,
        ],
    ) -> CitationValidationResult:
        """
        Valide directement un objet GenerationResponse.
        """

        return self.validate_answer(
            answer=response.answer,
            contexts=contexts,
        )