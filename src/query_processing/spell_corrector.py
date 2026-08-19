from __future__ import annotations

import re

from rapidfuzz import fuzz, process
from spellchecker import SpellChecker


class QuerySpellCorrector:
    """
    Correcteur orthographique orienté domaine Azure.

    Priorité :
    1. Acronymes techniques
    2. Termes métier Azure
    3. Dictionnaire anglais général

    Le but est d'éviter des erreurs comme :
        Kuburnit -> subunit
        AKc -> aka
    """

    def __init__(self) -> None:
        self.spell = SpellChecker(
            language="en",
            distance=2,
        )

        # ====================================================
        # Termes métier canoniques
        # ====================================================

        self.domain_terms: dict[str, str] = {
            "azure": "Azure",
            "cloud": "cloud",
            "computing": "computing",
            "databricks": "Databricks",
            "kubernetes": "Kubernetes",
            "devops": "DevOps",
            "entra": "Entra",
            "cosmos": "Cosmos",
            "cosmosdb": "CosmosDB",
            "serverless": "serverless",
            "finops": "FinOps",
            "reliability": "reliability",
            "resiliency": "resiliency",
            "availability": "availability",
            "scalability": "scalability",
            "security": "security",
            "governance": "governance",
            "architecture": "architecture",
            "workload": "workload",
            "storage": "storage",
            "container": "container",
            "containers": "containers",
            "monitor": "Monitor",
            "advisor": "Advisor",
        }

        # ====================================================
        # Acronymes techniques
        # ====================================================

        self.acronyms: dict[str, str] = {
            "aks": "AKS",
            "vm": "VM",
            "vms": "VMs",
            "ai": "AI",
            "ml": "ML",
            "rag": "RAG",
            "api": "API",
            "rbac": "RBAC",
            "sla": "SLA",
            "dr": "DR",
            "bc": "BC",
        }

        # ====================================================
        # Alias / erreurs fréquentes connues
        # ====================================================

        self.domain_aliases: dict[str, str] = {
            "clod": "cloud",
            "claud": "cloud",

            "computin": "computing",
            "compoting": "computing",

            "databrikc": "Databricks",
            "databrick": "Databricks",
            "databrik": "Databricks",

            "kuburnit": "Kubernetes",
            "kubernet": "Kubernetes",
            "kubernets": "Kubernetes",
            "kubernete": "Kubernetes",

            "reliabiliti": "reliability",
            "reliabilty": "reliability",

            "scalabiliti": "scalability",
            "scalabilty": "scalability",

            "azur": "Azure",
        }

        self.domain_vocabulary = set(
            self.domain_terms.keys()
        )

        self.spell.word_frequency.load_words(
            self.domain_vocabulary
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _is_word(
        token: str,
    ) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_-]*",
                token,
            )
        )

    @staticmethod
    def _restore_case(
        original: str,
        corrected: str,
    ) -> str:
        """
        Conserve la casse si aucune casse canonique particulière
        n'est nécessaire.
        """

        if original.isupper():
            return corrected.upper()

        if original[:1].isupper():
            return (
                corrected[:1].upper()
                + corrected[1:]
            )

        return corrected

    # ========================================================
    # Acronymes
    # ========================================================

    def _correct_acronym(
        self,
        word: str,
    ) -> str | None:
        normalized = word.lower()

        if normalized in self.acronyms:
            return self.acronyms[normalized]

        # Mot court : chercher d'abord parmi les acronymes
        if 2 <= len(normalized) <= 5:
            result = process.extractOne(
                normalized,
                self.acronyms.keys(),
                scorer=fuzz.ratio,
            )

            if result is not None:
                candidate, score, _ = result

                # Ex:
                # AKc -> AKS
                if score >= 65:
                    return self.acronyms[
                        candidate
                    ]

        return None

    # ========================================================
    # Termes métier
    # ========================================================

    def _correct_domain_term(
        self,
        word: str,
    ) -> str | None:
        normalized = word.lower()

        # Forme déjà correcte
        if normalized in self.domain_terms:
            return self.domain_terms[
                normalized
            ]

        # Alias explicitement connu
        if normalized in self.domain_aliases:
            return self.domain_aliases[
                normalized
            ]

        result = process.extractOne(
            normalized,
            self.domain_vocabulary,
            scorer=fuzz.WRatio,
        )

        if result is None:
            return None

        candidate, score, _ = result

        length = len(normalized)

        # Seuil adaptatif
        if length >= 8:
            threshold = 62
        elif length >= 5:
            threshold = 68
        else:
            threshold = 78

        if score >= threshold:
            return self.domain_terms[
                candidate
            ]

        return None

    # ========================================================
    # Correction générale
    # ========================================================

    def _correct_general_word(
        self,
        word: str,
    ) -> str:
        normalized = word.lower()

        if normalized in self.spell:
            return word

        candidate = self.spell.correction(
            normalized
        )

        if not candidate:
            return word

        return self._restore_case(
            original=word,
            corrected=candidate,
        )

    # ========================================================
    # Correction d'un token
    # ========================================================

    def _correct_word(
        self,
        word: str,
    ) -> str:
        normalized = word.lower()

        # ----------------------------------------------------
        # 1. Acronymes d'abord
        # ----------------------------------------------------

        acronym = self._correct_acronym(
            word
        )

        if acronym is not None:
            return acronym

        # ----------------------------------------------------
        # 2. Domaine Azure
        # ----------------------------------------------------

        domain_candidate = (
            self._correct_domain_term(
                word
            )
        )

        if domain_candidate is not None:
            return domain_candidate

        # ----------------------------------------------------
        # 3. Mots très courts non techniques
        # ----------------------------------------------------

        if len(normalized) <= 2:
            return word

        # ----------------------------------------------------
        # 4. Correcteur anglais général
        # ----------------------------------------------------

        return self._correct_general_word(
            word
        )

    # ========================================================
    # Normalisation d'une requête
    # ========================================================

    def normalize(
        self,
        query: str,
    ) -> str:
        normalized_query = str(
            query or ""
        ).strip()

        if not normalized_query:
            return normalized_query

        tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9_-]*|"
            r"\d+|"
            r"[^\w\s]",
            normalized_query,
        )

        corrected_tokens: list[str] = []

        for token in tokens:
            if self._is_word(
                token
            ):
                corrected_tokens.append(
                    self._correct_word(
                        token
                    )
                )
            else:
                corrected_tokens.append(
                    token
                )

        result = " ".join(
            corrected_tokens
        )

        # Ponctuation
        result = re.sub(
            r"\s+([?.!,;:])",
            r"\1",
            result,
        )

        result = re.sub(
            r"([(\[])\s+",
            r"\1",
            result,
        )

        result = re.sub(
            r"\s+([)\]])",
            r"\1",
            result,
        )

        result = re.sub(
            r"\s{2,}",
            " ",
            result,
        )

        return result.strip()