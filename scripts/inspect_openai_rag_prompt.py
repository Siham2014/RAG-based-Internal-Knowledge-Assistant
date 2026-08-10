from __future__ import annotations

from src.generation import (
    GenerationContext,
    GenerationRequest,
)
from src.generation.prompt_builder import (
    RAGPromptBuilder,
)


def build_test_request() -> GenerationRequest:
    """
    Reproduit exactement le petit test OpenAI utilisé
    pour valider le provider.

    Aucun appel API n'est réalisé.
    """

    context = GenerationContext(
        rank=1,
        source=(
            "well-architected__"
            "sustainability__overview.md"
        ),
        chunk_id=(
            "fixed__1024__"
            "well-architected__"
            "sustainability__overview.md__"
            "chunk_0"
        ),
        content=(
            "Sustainability for workloads in the cloud "
            "is a shared responsibility between the cloud "
            "provider and the organization."
        ),
        document_format="markdown",
        page_number=None,
        source_url=None,
        reranker_score=0.981162,
    )

    return GenerationRequest(
        question=(
            "Who shares responsibility for the "
            "sustainability of cloud workloads?"
        ),
        contexts=(
            context,
        ),
        language="en",
        max_output_tokens=60,
        temperature=0.0,
    )


def count_words(
    text: str,
) -> int:
    return len(
        text.split()
    )


def main() -> None:
    print("=" * 100)
    print("INSPECTION DU PROMPT RAG POUR OPENAI")
    print("=" * 100)
    print()
    print(
        "IMPORTANT : aucun appel OpenAI "
        "n'est effectué par ce script."
    )

    request = build_test_request()

    builder = RAGPromptBuilder()

    messages = builder.build_messages(
        request
    )

    print()
    print("=" * 100)
    print("QUESTION")
    print("=" * 100)
    print(request.question)

    print()
    print("=" * 100)
    print("MESSAGES CONSTRUITS")
    print("=" * 100)

    total_characters = 0
    total_words = 0

    for index, message in enumerate(
        messages,
        start=1,
    ):
        role = str(
            message.get(
                "role",
                "",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        character_count = len(
            content
        )

        word_count = count_words(
            content
        )

        total_characters += (
            character_count
        )

        total_words += (
            word_count
        )

        print()
        print("-" * 100)

        print(
            f"MESSAGE {index}"
        )

        print(
            "Role :",
            role,
        )

        print(
            "Caractères :",
            character_count,
        )

        print(
            "Mots :",
            word_count,
        )

        print("-" * 100)
        print(content)

    print()
    print("=" * 100)
    print("STATISTIQUES")
    print("=" * 100)

    print(
        "Nombre de messages :",
        len(messages),
    )

    print(
        "Caractères totaux :",
        total_characters,
    )

    print(
        "Mots totaux :",
        total_words,
    )

    # Estimation grossière uniquement.
    #
    # En anglais, un token représente souvent
    # environ 3 à 4 caractères selon le texte.
    approximate_tokens = round(
        total_characters / 4
    )

    print(
        "Tokens estimés approximativement :",
        approximate_tokens,
    )

    print()
    print("=" * 100)
    print("PARAMÈTRES DE GÉNÉRATION")
    print("=" * 100)

    print(
        "max_output_tokens demandé :",
        request.max_output_tokens,
    )

    print(
        "temperature :",
        request.temperature,
    )

    print()
    print("=" * 100)
    print("FIN DE L'INSPECTION")
    print("=" * 100)


if __name__ == "__main__":
    main()