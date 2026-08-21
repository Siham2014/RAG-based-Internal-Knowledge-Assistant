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

REFUSAL_MESSAGE_AR = "لا أعرف بناءً على المستندات الداخلية المتاحة."


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
                "If at least one supplied context directly answers "
                "the question, you must answer using that context "
                "and its authorized citation. "
                "An explicit enumeration in a context is direct evidence, "
                "even when the context introduces the listed items with "
                "different wording than the question. Synthesize only the "
                "items actually enumerated in that context. When a context "
                "states a requested count or category and later enumerates "
                "that same number of relevant items, use that enumeration "
                "as the answer; do not require a repeated label sentence. "
                "Do not use the refusal sentence when relevant "
                "evidence is present. "
                "If the available evidence is insufficient, "
                "answer exactly with this sentence and nothing else: "
                f"'{REFUSAL_MESSAGE_EN}'"
            )

        if normalized_language == "ar":
            return (
                "أنت مساعد معرفة داخلي. أجب فقط من السياقات الوثائقية المقدمة. "
                "لا تستخدم معرفة خارجية أو معلومات غير مدعومة. يجب دعم كل معلومة "
                "واقعية باستشهاد مصرح به واحد على الأقل، وانسخ الاستشهاد حرفياً "
                "دون ترجمته أو تعديله. إذا كانت الأدلة غير كافية، أجب فقط: "
                f"'{REFUSAL_MESSAGE_AR}'"
            )

        return (
            "Tu es un assistant documentaire interne. "
            "Réponds intégralement en français, même si la question de "
            "retrieval ou les contextes documentaires sont en anglais. "
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
            "Si au moins un contexte fourni répond directement "
            "à la question, tu dois répondre en utilisant ce "
            "contexte et sa citation autorisée. "
            "N'utilise pas la phrase de refus lorsque des preuves "
            "pertinentes sont présentes. "
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

        normalized_question = str(
            request.question or ""
        ).strip()

        if not normalized_question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if not request.contexts:
            raise ValueError(
                "Au moins un contexte documentaire est requis."
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

        normalized_language = (
            str(request.language)
            .strip()
            .lower()
        )

        if normalized_language == "en":
            return (
                "QUESTION\n"
                f"{normalized_question}\n\n"
                "DOCUMENTARY CONTEXTS\n"
                f"{contexts_text}\n\n"
                "FINAL INSTRUCTIONS\n"
                f"- {self._style_instruction(request.response_style, 'en')}\n"
                "- Use only information explicitly present "
                "in the contexts.\n"
                "- If at least one context contains direct evidence, "
                "answer the question.\n"
                "- Treat an explicit enumeration as direct evidence even "
                "when its introductory wording differs from the question; "
                "use only the items actually listed.\n"
                "- When a context states the requested count or category "
                "and later lists that same number of relevant items, answer "
                "from that list without requiring a repeated label.\n"
                "- Do not refuse when relevant evidence is present.\n"
                "- Place each citation immediately after "
                "the supported statement.\n"
                "- Copy the authorized citation exactly.\n"
                "- Do not write '[source: ...]'.\n"
                "- Do not modify the source name or chunk ID.\n"
                "- Do not create an external bibliography.\n"
                "- If the contexts do not contain enough evidence, "
                f"reply exactly: {REFUSAL_MESSAGE_EN}"
            )

        if normalized_language == "ar":
            return (
                "السؤال\n"
                f"{normalized_question}\n\n"
                "السياقات الوثائقية\n"
                f"{contexts_text}\n\n"
                "التعليمات النهائية\n"
                f"- {self._style_instruction(request.response_style, 'ar')}\n"
                "- استخدم فقط المعلومات الواردة صراحة في السياقات.\n"
                "- ضع كل استشهاد مباشرة بعد العبارة التي يدعمها.\n"
                "- انسخ الاستشهاد المصرح به حرفياً دون تعديل.\n"
                f"- إذا لم تكن الأدلة كافية، أجب حرفياً: {REFUSAL_MESSAGE_AR}"
            )

        return (
            "QUESTION\n"
            f"{normalized_question}\n\n"
            "CONTEXTES DOCUMENTAIRES\n"
            f"{contexts_text}\n\n"
            "INSTRUCTIONS FINALES\n"
            f"- {self._style_instruction(request.response_style, 'fr')}\n"
            "- Rédige toute la réponse en français, sauf les citations "
            "autorisées qui doivent rester strictement inchangées.\n"
            "- Utilise uniquement les informations explicitement "
            "présentes dans les contextes.\n"
            "- Si au moins un contexte contient une preuve directe, "
            "réponds à la question.\n"
            "- Ne refuse pas lorsque des preuves pertinentes "
            "sont présentes.\n"
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

    @staticmethod
    def _style_instruction(style: str, language: str) -> str:
        instructions = {
            "en": {
                "concise": "Keep the answer concise and direct.",
                "detailed": "Give a detailed explanation while staying within the evidence.",
                "expert": "Use precise expert terminology while staying within the evidence.",
            },
            "fr": {
                "concise": "Produis une réponse concise et directe.",
                "detailed": "Produis une explication détaillée sans dépasser les preuves.",
                "expert": "Utilise une terminologie experte sans dépasser les preuves.",
            },
            "ar": {
                "concise": "اجعل الإجابة موجزة ومباشرة.",
                "detailed": "قدم شرحاً مفصلاً ضمن حدود الأدلة.",
                "expert": "استخدم مصطلحات دقيقة بمستوى خبير ضمن حدود الأدلة.",
            },
        }
        return instructions.get(language, instructions["en"])[style]

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

        if not isinstance(
            request,
            GenerationRequest,
        ):
            raise TypeError(
                "request doit être une instance "
                "de GenerationRequest."
            )

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

        normalized_content = str(
            context.content or ""
        ).strip()

        if not normalized_content:
            raise ValueError(
                "Le contenu du contexte ne peut pas être vide."
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
            f"{normalized_content}"
        )
