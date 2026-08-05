from __future__ import annotations

from src.generation.models import (
    GenerationContext,
    GenerationRequest,
)


REFUSAL_MESSAGE_FR = (
    "Je ne sais pas répondre à cette question "
    "à partir des documents internes disponibles."
)

REFUSAL_MESSAGE_EN = (
    "I don't know based on the available internal documents."
)


class RAGPromptBuilder:
    """
    Construit un prompt strict et indépendant du fournisseur.
    """

    def build_system_prompt(
        self,
        language: str,
    ) -> str:
        normalized_language = (
            str(language)
            .strip()
            .lower()
        )

        if normalized_language == "en":
            return (
                "You are an internal knowledge assistant. "
                "Answer only from the supplied contexts. "
                "Do not use external knowledge. "
                "Every factual statement must be supported "
                "by at least one citation using exactly this "
                "format: [source:chunk_id]. "
                "Never invent a source or chunk identifier. "
                f"If the evidence is insufficient, answer: "
                f"'{REFUSAL_MESSAGE_EN}'"
            )

        return (
            "Tu es un assistant documentaire interne. "
            "Réponds uniquement à partir des contextes fournis. "
            "N'utilise aucune connaissance externe. "
            "Chaque affirmation factuelle doit être accompagnée "
            "d'au moins une citation au format exact "
            "[source:chunk_id]. "
            "N'invente jamais une source ou un identifiant de chunk. "
            "Lorsque les preuves sont insuffisantes, réponds : "
            f"« {REFUSAL_MESSAGE_FR} »"
        )

    def build_user_prompt(
        self,
        request: GenerationRequest,
    ) -> str:
        context_blocks = [
            self._format_context(
                context
            )
            for context in request.contexts
        ]

        contexts_text = "\n\n".join(
            context_blocks
        )

        return (
            "QUESTION\n"
            f"{request.question.strip()}\n\n"
            "CONTEXTES DOCUMENTAIRES\n"
            f"{contexts_text}\n\n"
            "INSTRUCTIONS\n"
            "- Produis une réponse claire et concise.\n"
            "- Utilise uniquement les informations des contextes.\n"
            "- Ajoute les citations immédiatement après "
            "les affirmations correspondantes.\n"
            "- Utilise uniquement les identifiants de citation "
            "fournis dans les contextes.\n"
            "- Ne crée aucune bibliographie externe."
        )

    def build_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    request.language
                ),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(
                    request
                ),
            },
        ]

    @staticmethod
    def _format_context(
        context: GenerationContext,
    ) -> str:
        source_url_line = (
            f"URL: {context.source_url}\n"
            if context.source_url
            else ""
        )

        page_line = (
            f"Page: {context.page_number}\n"
            if context.page_number is not None
            else ""
        )

        return (
            f"--- CONTEXTE {context.rank} ---\n"
            f"Citation autorisée: {context.citation_id}\n"
            f"Source: {context.source}\n"
            f"Chunk ID: {context.chunk_id}\n"
            f"{source_url_line}"
            f"{page_line}"
            f"Contenu:\n{context.content.strip()}"
        )