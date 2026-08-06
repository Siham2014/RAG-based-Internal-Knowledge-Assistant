from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.generation.models import (
    GenerationContext,
    GenerationResponse,
)


BRACKET_PATTERN = re.compile(
    r"\[([^\[\]]+)\]"
)


@dataclass(frozen=True)
class CitationValidationResult:
    """
    Résultat de la validation des citations.
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
    Vérifie que les citations générées correspondent
    aux contextes réellement transmis au modèle.

    Le validateur accepte :

    - la citation exacte ;
    - certains préfixes ajoutés par le modèle, comme
      ``source:`` ou ``citation:`` ;
    - certaines variations sûres de l'extension dans
      l'identifiant du chunk, comme ``__md__chunk_0``.

    Une citation n'est acceptée que si sa version normalisée
    correspond exactement à une citation autorisée.
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
    def authorized_citations(
        contexts: tuple[
            GenerationContext,
            ...,
        ],
    ) -> tuple[str, ...]:
        """
        Retourne les citations autorisées provenant
        des contextes transmis au LLM.
        """

        return tuple(
            context.citation_id
            for context in contexts
        )

    @staticmethod
    def _normalize_inner_text(
        inner_text: str,
    ) -> str:
        """
        Retire uniquement certains préfixes ajoutés
        par les modèles.

        Exemple :

            source: document.md:chunk_0

        devient :

            document.md:chunk_0
        """

        normalized = str(
            inner_text or ""
        ).strip()

        normalized = re.sub(
            r"^(?:source|citation|reference|ref)\s*:\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        return normalized.strip()

    @staticmethod
    def _canonicalize_citation(
        citation: str,
    ) -> str:
        """
        Normalise uniquement certaines variations sûres
        introduites par les modèles de langage.

        Exemple :

            overview__md__chunk_0

        devient :

            overview.md__chunk_0
        """

        normalized = str(
            citation or ""
        ).strip()

        replacements = {
            "__md__chunk_": ".md__chunk_",
            "__pdf__chunk_": ".pdf__chunk_",
            "__html__chunk_": ".html__chunk_",
            "__htm__chunk_": ".htm__chunk_",
            "__txt__chunk_": ".txt__chunk_",
        }

        for incorrect, correct in replacements.items():
            normalized = normalized.replace(
                incorrect,
                correct,
            )

        return normalized

    @classmethod
    def extract_citations(
        cls,
        answer: str,
    ) -> tuple[str, ...]:
        """
        Extrait les éléments entre crochets et normalise
        les préfixes de forme connus.
        """

        normalized_answer = str(
            answer or ""
        )

        citations: list[str] = []

        for match in BRACKET_PATTERN.finditer(
            normalized_answer
        ):
            inner_text = cls._normalize_inner_text(
                match.group(1)
            )

            if ":" not in inner_text:
                continue

            citation = f"[{inner_text}]"

            if citation not in citations:
                citations.append(
                    citation
                )

        return tuple(citations)

    def validate_answer(
        self,
        answer: str,
        contexts: tuple[
            GenerationContext,
            ...,
        ],
    ) -> CitationValidationResult:
        """
        Valide les citations présentes dans une réponse.
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

        raw_detected = self.extract_citations(
            answer
        )

        detected = tuple(
            self._canonicalize_citation(
                citation
            )
            for citation in raw_detected
        )

        authorized_ordered = (
            self.authorized_citations(
                contexts
            )
        )

        authorized_set = set(
            authorized_ordered
        )

        valid_citations = tuple(
            citation
            for citation in detected
            if citation in authorized_set
        )

        invalid_citations = tuple(
            citation
            for citation in detected
            if citation not in authorized_set
        )

        if (
            self.require_at_least_one_citation
            and not detected
        ):
            return CitationValidationResult(
                valid=False,
                detected_citations=(),
                valid_citations=(),
                invalid_citations=(),
                missing_citations=(
                    authorized_ordered
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
                invalid_citations=(
                    invalid_citations
                ),
                missing_citations=(),
                reason=(
                    "La réponse contient une ou plusieurs "
                    "citations qui ne correspondent à aucun "
                    "contexte autorisé."
                ),
            )

        if (
            self.require_at_least_one_citation
            and not valid_citations
        ):
            return CitationValidationResult(
                valid=False,
                detected_citations=detected,
                valid_citations=(),
                invalid_citations=detected,
                missing_citations=(
                    authorized_ordered
                ),
                reason=(
                    "Aucune citation valide n'a été trouvée."
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
                "correspondent aux contextes autorisés."
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
        Valide directement une GenerationResponse.
        """

        if not isinstance(
            response,
            GenerationResponse,
        ):
            raise TypeError(
                "response doit être une instance "
                "de GenerationResponse."
            )

        return self.validate_answer(
            answer=response.answer,
            contexts=contexts,
        )