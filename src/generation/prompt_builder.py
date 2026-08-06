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
    Construit un prompt RAG strict et indépendant
    du fournisseur LLM.
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
                "Answer only from the supplied documentary contexts. "
                "Do not use external knowledge, assumptions, "
                "or unsupported information. "
                "Every factual statement must be supported "
                "by at least one authorized citation. "
                "Copy each authorized citation exactly as it appears "
                "in the corresponding context. "
                "Do not add the words 'source', 'citation', "
                "'reference', or any other prefix inside brackets. "
                "Do not add spaces inside the citation. "
                "Do not shorten, translate, rewrite, or invent "
                "a source name or chunk identifier. "
                "An authorized citation may look like: "
                "[document.md:fixed__document.md__chunk_0]. "
                "If the available evidence is insufficient, "
                "answer exactly with this sentence and nothing else: "
                f"'{REFUSAL_MESSAGE_EN}'"
            )

        return (
            "Tu es un assistant documentaire interne. "
            "Réponds uniquement à partir des contextes "
            "documentaires fournis. "
            "N'utilise aucune connaissance externe, supposition "
            "ou information non démontrée. "
            "Chaque affirmation factuelle doit être accompagnée "
            "d'au moins une citation autorisée. "
            "Copie chaque citation autorisée exactement telle "
            "qu'elle apparaît dans le contexte correspondant. "
            "N'ajoute pas les mots « source », « citation », "
            "« référence » ni aucun autre préfixe entre crochets. "
            "N'ajoute aucun espace à l'intérieur de la citation. "
            "Ne raccourcis, ne traduis, ne reformule et n'invente "
            "jamais un nom de source ou un identifiant de chunk. "
            "Une citation autorisée peut ressembler à : "
            "[document.md:fixed__document.md__chunk_0]. "
            "Si les preuves disponibles sont insuffisantes, "
            "réponds exactement avec cette phrase et rien d'autre : "
            f"« {REFUSAL_MESSAGE_FR} »"
        )

    def build_user_prompt(
        self,
        request: GenerationRequest,
    ) -> str:
        if not isinstance(
            request,
            GenerationRequest,
        ):
            raise TypeError(
                "request doit être une instance "
                "de GenerationRequest."
            )

        context_blocks = [
            self._format_context(
                context
            )
            for context in request.contexts
        ]

        contexts_text = "\n\n".join(
            context_blocks
        )

        if request.language.strip().lower() == "en":
            return (
                "QUESTION\n"
                f"{request.question.strip()}\n\n"
                "DOCUMENTARY CONTEXTS\n"
                f"{contexts_text}\n\n"
                "FINAL INSTRUCTIONS\n"
                "- Give a clear and concise answer.\n"
                "- Use only information explicitly present "
                "in the contexts.\n"
                "- Place each citation immediately after "
                "the supported statement.\n"
                "- Copy the authorized citation exactly.\n"
                "- Do not write '[source: ...]'.\n"
                "- Do not modify the source name or chunk ID.\n"
                "- Do not create an external bibliography.\n"
                "- If the contexts do not contain enough evidence, "
                f"reply exactly: {REFUSAL_MESSAGE_EN}"
            )

        return (
            "QUESTION\n"
            f"{request.question.strip()}\n\n"
            "CONTEXTES DOCUMENTAIRES\n"
            f"{contexts_text}\n\n"
            "INSTRUCTIONS FINALES\n"
            "- Produis une réponse claire et concise.\n"
            "- Utilise uniquement les informations explicitement "
            "présentes dans les contextes.\n"
            "- Place chaque citation immédiatement après "
            "l'affirmation qu'elle justifie.\n"
            "- Copie exactement la citation autorisée.\n"
            "- N'écris pas « [source: ...] ».\n"
            "- Ne modifie ni le nom de la source ni "
            "l'identifiant du chunk.\n"
            "- Ne crée aucune bibliographie externe.\n"
            "- Si les contextes ne contiennent pas assez de preuves, "
            f"réponds exactement : {REFUSAL_MESSAGE_FR}"
        )

    def build_messages(
        self,
        request: GenerationRequest,
    ) -> list[dict[str, str]]:
        """
        Construit les messages standards.

        La compatibilité avec les modèles qui ne prennent pas
        en charge le rôle system, comme certains modèles Gemma,
        est gérée dans HuggingFaceProvider._prepare_messages().
        """

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
        if not isinstance(
            context,
            GenerationContext,
        ):
            raise TypeError(
                "context doit être une instance "
                "de GenerationContext."
            )

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

        format_line = (
            f"Format: {context.document_format}\n"
            if context.document_format
            else ""
        )

        score_line = (
            f"Reranker score: {context.reranker_score:.6f}\n"
            if context.reranker_score is not None
            else ""
        )

        return (
            f"--- CONTEXT {context.rank} ---\n"
            "AUTHORIZED CITATION — COPY EXACTLY:\n"
            f"{context.citation_id}\n"
            f"Source: {context.source}\n"
            f"Chunk ID: {context.chunk_id}\n"
            f"{format_line}"
            f"{source_url_line}"
            f"{page_line}"
            f"{score_line}"
            "Content:\n"
            f"{context.content.strip()}"
        )