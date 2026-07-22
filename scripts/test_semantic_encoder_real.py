from __future__ import annotations

import numpy as np

from src.chunking.semantic_encoder import (
    SentenceTransformerEncoder,
    cosine_similarity,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def separator() -> None:
    print("=" * 70)


def test_model_loading() -> SentenceTransformerEncoder:
    """
    Vérifie que le modèle SentenceTransformer peut être chargé.
    """

    separator()
    print("TEST DU CHARGEMENT DU MODELE SEMANTIQUE")
    separator()

    encoder = SentenceTransformerEncoder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=8,
        normalize_embeddings=True,
    )

    print("Modèle configuré :", encoder.model_name)
    print("Device :", encoder.device)
    print("Batch size :", encoder.batch_size)

    assert encoder.model_name == MODEL_NAME
    assert encoder.batch_size == 8

    print("Configuration du modèle : OK")

    return encoder


def test_real_embeddings(
    encoder: SentenceTransformerEncoder,
) -> np.ndarray:
    """
    Vérifie la génération réelle des embeddings.
    """

    print()
    separator()
    print("TEST DE GENERATION DES EMBEDDINGS")
    separator()

    sentences = [
        "Azure availability zones improve application reliability.",
        "Redundancy helps applications tolerate infrastructure failures.",
        "Cloud billing reports show monthly service costs.",
        "Budgets help organizations control Azure expenses.",
    ]

    embeddings = encoder.encode(sentences)

    print("Nombre de phrases :", len(sentences))
    print("Forme de la matrice :", embeddings.shape)
    print("Type des données :", embeddings.dtype)

    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(sentences)
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32

    print("Dimension d'embedding :", embeddings.shape[1])
    print("Génération des embeddings : OK")

    return embeddings


def test_normalization(
    embeddings: np.ndarray,
) -> None:
    """
    Vérifie que chaque embedding possède une norme proche de 1.
    """

    print()
    separator()
    print("TEST DE NORMALISATION")
    separator()

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    for index, norm in enumerate(norms):
        print(
            f"Embedding {index} : "
            f"norme={float(norm):.6f}"
        )

    assert np.allclose(
        norms,
        np.ones_like(norms),
        atol=1e-5,
    )

    print("Normalisation des embeddings : OK")


def test_semantic_similarity(
    embeddings: np.ndarray,
) -> None:
    """
    Compare les similarités entre les phrases.

    Les deux premières phrases concernent la fiabilité.
    Les deux dernières concernent les coûts.
    """

    print()
    separator()
    print("TEST DES SIMILARITES SEMANTIQUES")
    separator()

    reliability_similarity = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    topic_change_similarity = cosine_similarity(
        embeddings[1],
        embeddings[2],
    )

    cost_similarity = cosine_similarity(
        embeddings[2],
        embeddings[3],
    )

    print(
        "Fiabilité ↔ Fiabilité :",
        round(reliability_similarity, 4),
    )

    print(
        "Fiabilité ↔ Coût :",
        round(topic_change_similarity, 4),
    )

    print(
        "Coût ↔ Coût :",
        round(cost_similarity, 4),
    )

    assert reliability_similarity > topic_change_similarity
    assert cost_similarity > topic_change_similarity

    print("Détection du changement de sujet : OK")


def test_deterministic_embeddings(
    encoder: SentenceTransformerEncoder,
) -> None:
    """
    Vérifie que le même texte produit le même embedding.
    """

    print()
    separator()
    print("TEST DES EMBEDDINGS DETERMINISTES")
    separator()

    sentence = [
        "Azure reliability requires redundancy and monitoring."
    ]

    first_embedding = encoder.encode(sentence)
    second_embedding = encoder.encode(sentence)

    assert np.allclose(
        first_embedding,
        second_embedding,
        atol=1e-6,
    )

    print("Embeddings déterministes : OK")


def test_empty_input(
    encoder: SentenceTransformerEncoder,
) -> None:
    """
    Vérifie la gestion d'une liste vide.
    """

    print()
    separator()
    print("TEST D'UNE LISTE VIDE")
    separator()

    embeddings = encoder.encode([])

    print(
        "Forme retournée :",
        embeddings.shape,
    )

    assert embeddings.shape == (0, 0)

    print("Liste vide correctement gérée : OK")


def test_invalid_empty_text(
    encoder: SentenceTransformerEncoder,
) -> None:
    """
    Vérifie qu'une phrase vide est refusée.
    """

    print()
    separator()
    print("TEST D'UN TEXTE VIDE")
    separator()

    try:
        encoder.encode(
            [
                "Azure reliability",
                "",
            ]
        )

    except Exception as error:
        print(
            "Texte vide détecté :",
            error,
        )

    else:
        raise AssertionError(
            "Une phrase vide aurait dû être refusée."
        )

    print("Validation des textes : OK")


def main() -> None:
    encoder = test_model_loading()

    embeddings = test_real_embeddings(
        encoder
    )

    test_normalization(
        embeddings
    )

    test_semantic_similarity(
        embeddings
    )

    test_deterministic_embeddings(
        encoder
    )

    test_empty_input(
        encoder
    )

    test_invalid_empty_text(
        encoder
    )

    print()
    separator()
    print("ENCODEUR SEMANTIQUE REEL FONCTIONNEL.")
    separator()


if __name__ == "__main__":
    main()