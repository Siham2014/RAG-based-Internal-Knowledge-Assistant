from __future__ import annotations

from openai import OpenAI

from src.common.settings import get_settings


class QueryNormalizer:
    """
    Normalise une requête utilisateur avant le retrieval.

    Objectifs :
    - corriger les fautes d'orthographe ;
    - corriger les fautes de frappe ;
    - améliorer légèrement la grammaire ;
    - remettre en ordre une formulation maladroite ;
    - préserver strictement l'intention ;
    - préserver les noms propres, acronymes et termes techniques ;
    - ne jamais répondre à la question.
    """

    SYSTEM_PROMPT = """
You are a query normalization component for a document retrieval system.

Your ONLY task is to rewrite the user's query into a clean search query.

Rules:
1. Correct spelling and obvious typing errors.
2. Correct obvious grammatical errors when necessary.
3. Preserve the exact meaning and intent of the user.
4. Preserve technical terms, product names, proper nouns, acronyms,
   numbers, versions, identifiers, and domain-specific terminology.
5. Do NOT add information that the user did not provide.
6. Do NOT answer the question.
7. Do NOT explain your corrections.
8. Do NOT add quotation marks.
9. If the query is already correct, return it unchanged.
10. Return ONLY the normalized query.

Examples:

User:
what is clod computin

Output:
what is cloud computing

User:
How cn I improve Azure reliabiliti?

Output:
How can I improve Azure reliability?

User:
explan kubernetes archtectur

Output:
explain Kubernetes architecture

User:
What is AKS?

Output:
What is AKS?
""".strip()

    def __init__(self) -> None:
    settings = get_settings()

    api_key = settings.openai_api_key

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    self.client = OpenAI(
        api_key=api_key,
    )

    self.model = (
        settings.generation.model_name
        or "gpt-5-nano"
    )

    def normalize(
        self,
        query: str,
    ) -> str:
        original = str(
            query or ""
        ).strip()

        if not original:
            return original

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.SYSTEM_PROMPT,
                input=original,
                reasoning={
                    "effort": "minimal",
                },
                max_output_tokens=80,
            )

            normalized = (
                response.output_text
                or ""
            ).strip()

            # Sécurité :
            # si le provider ne retourne rien,
            # conserver la requête originale.
            if not normalized:
                return original

            # Éviter une sortie anormalement longue.
            if len(normalized) > (
                max(len(original) * 3, 500)
            ):
                return original

            return normalized

        except Exception as error:
            # Le normalizer ne doit JAMAIS casser le RAG.
            print(
                "[QueryNormalizer] "
                f"Normalization failed: "
                f"{type(error).__name__}: {error}"
            )

            return original