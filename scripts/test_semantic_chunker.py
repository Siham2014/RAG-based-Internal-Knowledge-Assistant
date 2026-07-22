from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    TextBlock,
)
from src.chunking.registry import ChunkerRegistry
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.semantic_encoder import BaseSemanticEncoder
from src.chunking.tokenizer import WhitespaceTokenizer


class KeywordTestEncoder(BaseSemanticEncoder):
    """
    Encodeur déterministe utilisé uniquement dans les tests.

    Les textes liés à la fiabilité reçoivent un premier vecteur.
    Les textes liés aux coûts reçoivent un deuxième vecteur.
    Les autres textes reçoivent un troisième vecteur.

    Cela permet de tester le changement sémantique sans télécharger
    un modèle SentenceTransformer.
    """

    @property
    def model_name(self) -> str:
        return "keyword-test-encoder"

    def encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        vectors: list[list[float]] = []

        for text in texts:
            lowered = text.lower()

            if any(
                word in lowered
                for word in (
                    "reliability",
                    "availability",
                    "zone",
                    "failure",
                    "redundancy",
                )
            ):
                vectors.append(
                    [1.0, 0.0, 0.0]
                )

            elif any(
                word in lowered
                for word in (
                    "billing",
                    "cost",
                    "invoice",
                    "price",
                    "budget",
                )
            ):
                vectors.append(
                    [0.0, 1.0, 0.0]
                )

            else:
                vectors.append(
                    [0.0, 0.0, 1.0]
                )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


class TestSemanticChunker(SemanticChunker):
    """
    SemanticChunker utilisé uniquement pour tester le registry.

    Le registry transmet seulement la configuration au constructeur.
    Cette classe fournit donc automatiquement l'encodeur et le tokenizer
    utilisés pendant les tests.
    """

    def __init__(
        self,
        config: ChunkingConfig,
    ) -> None:
        super().__init__(
            config=config,
            encoder=KeywordTestEncoder(),
            tokenizer=WhitespaceTokenizer(),
        )


def separator() -> None:
    print("=" * 70)


def create_config(
    *,
    chunk_size: int = 30,
    chunk_overlap: int = 3,
    threshold: float = 0.65,
) -> ChunkingConfig:
    """
    Crée une configuration SemanticChunker commune aux tests.
    """

    return ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=1,
        tokenizer_name="whitespace",
        semantic_similarity_threshold=threshold,
    )


def create_blocks() -> list[TextBlock]:
    """
    Crée un document contenant deux sujets distincts :

    1. fiabilité Azure ;
    2. gestion des coûts Azure.
    """

    return [
        TextBlock.create(
            document_id="semantic-test",
            block_type="heading",
            text="Azure reliability",
            block_index=0,
            section_id="reliability",
            section_heading="Reliability",
            heading_path=[
                "Azure",
                "Reliability",
            ],
            document_title="Azure architecture",
        ),
        TextBlock.create(
            document_id="semantic-test",
            block_type="paragraph",
            text=(
                "Availability zones improve reliability. "
                "Redundancy protects applications during failures."
            ),
            block_index=1,
            section_id="reliability",
            section_heading="Reliability",
            heading_path=[
                "Azure",
                "Reliability",
            ],
            document_title="Azure architecture",
        ),
        TextBlock.create(
            document_id="semantic-test",
            block_type="heading",
            text="Cost management",
            block_index=2,
            section_id="cost",
            section_heading="Cost management",
            heading_path=[
                "Azure",
                "Cost management",
            ],
            document_title="Azure architecture",
        ),
        TextBlock.create(
            document_id="semantic-test",
            block_type="paragraph",
            text=(
                "Billing reports show monthly cloud costs. "
                "Budgets help teams control invoices."
            ),
            block_index=3,
            section_id="cost",
            section_heading="Cost management",
            heading_path=[
                "Azure",
                "Cost management",
            ],
            document_title="Azure architecture",
        ),
    ]


def create_chunker(
    *,
    chunk_size: int = 30,
    chunk_overlap: int = 3,
    threshold: float = 0.65,
) -> SemanticChunker:
    """
    Crée directement un SemanticChunker avec les composants de test.
    """

    return SemanticChunker(
        config=create_config(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            threshold=threshold,
        ),
        encoder=KeywordTestEncoder(),
        tokenizer=WhitespaceTokenizer(),
    )


def test_semantic_boundary() -> None:
    """
    Vérifie qu'un changement de sujet crée une frontière sémantique.
    """

    separator()
    print("TEST DU CHANGEMENT SEMANTIQUE")
    separator()

    chunker = create_chunker()

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    print(
        "Nombre de chunks :",
        len(chunks),
    )

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index} | "
            f"tokens={chunk.token_count} | "
            f"raison={chunk.metadata['boundary_reason']}"
        )

        print(
            "Texte :",
            chunk.text,
        )

        print("-" * 70)

        assert (
            chunk.strategy
            == ChunkingStrategy.SEMANTIC
        )

        assert chunk.token_count <= 30

        assert (
            chunk.metadata["semantic_encoder"]
            == "keyword-test-encoder"
        )

        assert (
            chunk.metadata[
                "semantic_similarity_threshold"
            ]
            == 0.65
        )

    assert len(chunks) >= 2

    assert (
        "reliability"
        in chunks[0].text.lower()
    )

    assert any(
        "billing" in chunk.text.lower()
        for chunk in chunks[1:]
    )

    assert (
        chunks[0].metadata["boundary_reason"]
        == "semantic_threshold"
    )

    print(
        "Frontière sémantique : OK"
    )


def test_size_limit() -> None:
    """
    Vérifie que chunk_size reste toujours une limite maximale.
    """

    print()
    separator()
    print("TEST DE LA TAILLE MAXIMALE")
    separator()

    text = " ".join(
        f"reliability_{index}"
        for index in range(27)
    )

    block = TextBlock.create(
        document_id="semantic-size-test",
        block_type="paragraph",
        text=text,
        block_index=0,
    )

    chunker = create_chunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    chunks = chunker.chunk_blocks(
        [block]
    )

    print(
        "Nombre de chunks :",
        len(chunks),
    )

    for chunk in chunks:
        print(
            chunk.chunk_index,
            chunk.token_count,
            chunk.text,
        )

        assert chunk.token_count <= 10

    assert len(chunks) >= 3

    print(
        "Taille maximale : OK"
    )


def test_overlap_metadata() -> None:
    """
    Vérifie le contenu réel et les métadonnées de l'overlap.
    """

    print()
    separator()
    print("TEST DE L'OVERLAP")
    separator()

    tokenizer = WhitespaceTokenizer()

    chunker = create_chunker(
        chunk_size=12,
        chunk_overlap=2,
    )

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    overlap_found = False

    for index, chunk in enumerate(chunks):
        overlap = chunk.metadata[
            "overlap_token_count"
        ]

        print(
            f"Chunk {index}: "
            f"overlap_token_count={overlap}"
        )

        assert 0 <= overlap <= 2

        if index == 0:
            assert overlap == 0
            continue

        if overlap > 0:
            overlap_found = True

            previous_suffix = (
                tokenizer.take_last_tokens(
                    chunks[index - 1].text,
                    overlap,
                )
            )

            current_prefix = (
                tokenizer.take_first_tokens(
                    chunk.text,
                    overlap,
                )
            )

            print(
                "Fin du chunk précédent :",
                previous_suffix,
            )

            print(
                "Début du chunk courant :",
                current_prefix,
            )

            assert (
                previous_suffix
                == current_prefix
            )

    assert overlap_found

    print(
        "Overlap sémantique : OK"
    )


def test_deterministic_ids() -> None:
    """
    Vérifie que le même document produit toujours les mêmes chunk IDs.
    """

    print()
    separator()
    print(
        "TEST DES IDENTIFIANTS DETERMINISTES"
    )
    separator()

    first_chunks = (
        create_chunker().chunk_blocks(
            create_blocks()
        )
    )

    second_chunks = (
        create_chunker().chunk_blocks(
            create_blocks()
        )
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first_chunks
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_chunks
    ]

    assert first_ids == second_ids

    print(
        "Identifiants déterministes : OK"
    )


def test_registry() -> None:
    """
    Vérifie la création du chunker à travers ChunkerRegistry.
    """

    print()
    separator()
    print("TEST DU REGISTRY")
    separator()

    ChunkerRegistry.register(
        ChunkingStrategy.SEMANTIC,
        TestSemanticChunker,
        replace=True,
    )

    chunker = ChunkerRegistry.create(
        create_config()
    )

    assert isinstance(
        chunker,
        TestSemanticChunker,
    )

    assert isinstance(
        chunker,
        SemanticChunker,
    )

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    assert chunks

    print(
        "Stratégies enregistrées :",
        ChunkerRegistry.available_strategies(),
    )

    print(
        "Chunker créé :",
        chunker,
    )

    print(
        "Nombre de chunks :",
        len(chunks),
    )

    print(
        "Création depuis le registre : OK"
    )


def test_invalid_threshold() -> None:
    """
    Vérifie qu'une configuration invalide est rejetée.
    """

    print()
    separator()
    print("TEST DU SEUIL INVALIDE")
    separator()

    try:
        create_config(
            threshold=1.5,
        )

    except Exception as error:
        print(
            "Seuil invalide détecté :",
            error,
        )

    else:
        raise AssertionError(
            "Le seuil invalide aurait dû être refusé."
        )

    print(
        "Validation du seuil : OK"
    )


def main() -> None:
    test_semantic_boundary()
    test_size_limit()
    test_overlap_metadata()
    test_deterministic_ids()
    test_registry()
    test_invalid_threshold()

    print()
    separator()
    print(
        "SEMANTIC CHUNKER FONCTIONNEL."
    )
    separator()


if __name__ == "__main__":
    main()