from __future__ import annotations

from src.chunking.tokenizer import (
    TokenizerError,
    WhitespaceTokenizer,
)


def print_separator() -> None:
    print("=" * 70)


def build_test_text(word_count: int) -> str:
    return " ".join(
        f"token_{index}"
        for index in range(word_count)
    )


def test_encode_decode() -> None:
    print_separator()
    print("TEST ENCODE / DECODE")
    print_separator()

    tokenizer = WhitespaceTokenizer()

    text = "Azure reliability improves application resilience"

    token_ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(token_ids)

    print("Tokenizer :", tokenizer.name)
    print("Texte :", text)
    print("Token IDs :", token_ids)
    print("Nombre de tokens :", tokenizer.count_tokens(text))
    print("Texte reconstruit :", reconstructed)

    assert len(token_ids) == 5
    assert reconstructed == text
    assert tokenizer.count_tokens(text) == 5


def test_windows_without_overlap() -> None:
    print()
    print_separator()
    print("TEST DES FENETRES SANS OVERLAP")
    print_separator()

    tokenizer = WhitespaceTokenizer()
    text = build_test_text(20)

    windows = tokenizer.split_token_windows(
        text,
        chunk_size=8,
        chunk_overlap=0,
    )

    print("Nombre total de tokens :", tokenizer.count_tokens(text))
    print("Nombre de fenêtres :", len(windows))

    for index, window in enumerate(windows):
        decoded = tokenizer.decode(window.token_ids)

        print(
            f"Fenêtre {index}: "
            f"start={window.start_token}, "
            f"end={window.end_token}, "
            f"count={window.token_count}"
        )

        print("Texte :", decoded)

    assert len(windows) == 3

    assert windows[0].start_token == 0
    assert windows[0].end_token == 8
    assert windows[0].token_count == 8

    assert windows[1].start_token == 8
    assert windows[1].end_token == 16
    assert windows[1].token_count == 8

    assert windows[2].start_token == 16
    assert windows[2].end_token == 20
    assert windows[2].token_count == 4


def test_windows_with_overlap() -> None:
    print()
    print_separator()
    print("TEST DES FENETRES AVEC OVERLAP")
    print_separator()

    tokenizer = WhitespaceTokenizer()
    text = build_test_text(20)

    windows = tokenizer.split_token_windows(
        text,
        chunk_size=8,
        chunk_overlap=2,
    )

    print("Nombre total de tokens :", tokenizer.count_tokens(text))
    print("Chunk size : 8")
    print("Overlap : 2")
    print("Step : 6")
    print("Nombre de fenêtres :", len(windows))

    for index, window in enumerate(windows):
        print(
            f"Fenêtre {index}: "
            f"start={window.start_token}, "
            f"end={window.end_token}, "
            f"count={window.token_count}"
        )

        print("Texte :", tokenizer.decode(window.token_ids))

    assert len(windows) == 3

    assert windows[0].start_token == 0
    assert windows[0].end_token == 8

    assert windows[1].start_token == 6
    assert windows[1].end_token == 14

    assert windows[2].start_token == 12
    assert windows[2].end_token == 20

    first_tail = windows[0].token_ids[-2:]
    second_head = windows[1].token_ids[:2]

    assert first_tail == second_head

    second_tail = windows[1].token_ids[-2:]
    third_head = windows[2].token_ids[:2]

    assert second_tail == third_head

    print()
    print("Overlap entre les fenêtres : OK")


def test_token_helpers() -> None:
    print()
    print_separator()
    print("TEST DES FONCTIONS UTILITAIRES")
    print_separator()

    tokenizer = WhitespaceTokenizer()

    text = "one two three four five six seven eight"

    first = tokenizer.take_first_tokens(text, 3)
    last = tokenizer.take_last_tokens(text, 3)
    truncated = tokenizer.truncate(text, 5)

    print("Premiers 3 tokens :", first)
    print("Derniers 3 tokens :", last)
    print("Texte tronqué à 5 tokens :", truncated)

    assert first == "one two three"
    assert last == "six seven eight"
    assert truncated == "one two three four five"


def test_empty_text() -> None:
    print()
    print_separator()
    print("TEST D'UN TEXTE VIDE")
    print_separator()

    tokenizer = WhitespaceTokenizer()

    assert tokenizer.encode("") == []
    assert tokenizer.count_tokens("") == 0

    windows = tokenizer.split_token_windows(
        "",
        chunk_size=256,
        chunk_overlap=32,
    )

    assert windows == []

    print("Gestion du texte vide : OK")


def test_invalid_parameters() -> None:
    print()
    print_separator()
    print("TEST DES PARAMETRES INVALIDES")
    print_separator()

    tokenizer = WhitespaceTokenizer()

    invalid_cases = [
        {
            "chunk_size": 0,
            "chunk_overlap": 0,
        },
        {
            "chunk_size": 256,
            "chunk_overlap": -1,
        },
        {
            "chunk_size": 256,
            "chunk_overlap": 256,
        },
        {
            "chunk_size": 256,
            "chunk_overlap": 300,
        },
    ]

    for case in invalid_cases:
        try:
            tokenizer.split_token_windows(
                "Azure reliability",
                chunk_size=case["chunk_size"],
                chunk_overlap=case["chunk_overlap"],
            )
        except TokenizerError as exc:
            print(
                f"Erreur détectée pour "
                f"size={case['chunk_size']}, "
                f"overlap={case['chunk_overlap']} : {exc}"
            )
        else:
            raise AssertionError(
                f"Une erreur était attendue pour {case}."
            )


def main() -> None:
    test_encode_decode()
    test_windows_without_overlap()
    test_windows_with_overlap()
    test_token_helpers()
    test_empty_text()
    test_invalid_parameters()

    print()
    print_separator()
    print("TOKENIZER COMMUN FONCTIONNEL.")
    print_separator()


if __name__ == "__main__":
    main()