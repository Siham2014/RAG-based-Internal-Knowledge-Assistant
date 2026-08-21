from __future__ import annotations

from types import SimpleNamespace

from src.conversation import (
    ConversationMessage,
    ConversationQueryRewriter,
    ConversationalRAGService,
    InMemoryConversationMemory,
    MessageRole,
)
from src.query_processing import IntentRouter, LanguageDetector, QueryIntent


class FakeProvider:
    provider_name = "fake"
    model_name = "fake-small-model"

    def complete_text(self, **kwargs: object) -> str:
        text = str(kwargs["user_input"])
        if "more explanation" in text:
            return "Explain cloud computing in more detail."
        if "Qu'est-ce" in text:
            return "What is cloud computing?"
        if "الحوسبة" in text:
            return "What is cloud computing?"
        return "What is cloud computing?"


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.last_query_normalization = None

    def answer(self, question: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append((question, kwargs))
        self.last_query_normalization = SimpleNamespace(normalized_query=question)
        return SimpleNamespace(answer=f"answer for: {question}")


def main() -> None:
    detector = LanguageDetector()
    assert detector.detect("What is cloud computing?").language == "en"
    assert detector.detect("Qu'est-ce que le cloud computing ?").language == "fr"
    assert detector.detect(
        "Quels sont les cinq piliers du cadre Azure Well-Architected ?"
    ).language == "fr"
    assert detector.detect("ما هي الحوسبة السحابية؟").language == "ar"

    router = IntentRouter()
    for greeting in ("Hi!", "bonjour", "مرحبا", "السلام عليكم"):
        assert router.route(greeting, False) is QueryIntent.GREETING
    assert router.route("more explanation", True) is QueryIntent.FOLLOW_UP
    assert router.route("What is cloud computing?", False) is QueryIntent.KNOWLEDGE_QUERY

    memory = InMemoryConversationMemory(max_conversations=10, max_messages=4)
    rag = FakeRAG()
    service = ConversationalRAGService(
        rag_pipeline=rag,  # type: ignore[arg-type]
        memory=memory,
        query_rewriter=ConversationQueryRewriter(FakeProvider()),
    )

    greeting = service.process("bonjour", conversation_id="greeting", language="auto")
    assert greeting.rag_response is None
    assert greeting.detected_language == "fr"
    assert rag.calls == []

    memory.append_message(
        "followup",
        ConversationMessage.create(MessageRole.USER, "What is cloud computing?"),
    )
    followup = service.process(
        "more explanation",
        conversation_id="followup",
        language="auto",
        response_style="detailed",
    )
    assert followup.intent is QueryIntent.FOLLOW_UP
    assert followup.rewritten_query == "Explain cloud computing in more detail."
    assert rag.calls[-1][0] == followup.rewritten_query
    assert rag.calls[-1][1]["normalize_query"] is False
    assert rag.calls[-1][1]["response_style"] == "detailed"

    french = service.process(
        "Qu'est-ce que le cloud computing ?",
        conversation_id="fr",
        language="auto",
    )
    assert french.reply_language == "fr"
    assert french.rewritten_query == "What is cloud computing?"

    arabic = service.process(
        "ما هي الحوسبة السحابية؟",
        conversation_id="ar",
        language="auto",
    )
    assert arabic.reply_language == "ar"
    assert arabic.rewritten_query == "What is cloud computing?"

    manual = service.process(
        "What is cloud computing?",
        conversation_id="manual",
        language="en",
        reply_language="fr",
    )
    assert manual.reply_language == "fr"
    assert rag.calls[-1][1]["reply_language"] == "fr"

    print("All conversational-layer tests passed.")


if __name__ == "__main__":
    main()
