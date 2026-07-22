from __future__ import annotations

from typing import Sequence

from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentChunk,
    TextBlock,
)
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.semantic_encoder import (
    SentenceTransformerEncoder,
)
from src.chunking.tokenizer import WhitespaceTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def separator() -> None:
    print("=" * 70)


def create_config(
    *,
    chunk_size: int = 40,
    chunk_overlap: int = 4,
    threshold: float = 0.45,
) -> ChunkingConfig:
    """
    Crée une configuration pour le SemanticChunker réel.
    """

    return ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=1,
        tokenizer_name="whitespace",
        preserve_sentences=True,
        preserve_code_blocks=True,
        preserve_tables=True,
        include_heading_context=True,
        semantic_similarity_threshold=threshold,
        metadata={
            "test_type": "real_semantic_chunker",
            "embedding_model": MODEL_NAME,
        },
    )


def create_blocks() -> list[TextBlock]:
    """
    Crée un document Azure avec trois sujets différents :

    1. fiabilité ;
    2. gestion des coûts ;
    3. sécurité des identités.
    """

    document_id = "azure-semantic-real-test"
    document_title = "Azure architecture guide"
    source_url = "https://learn.microsoft.com/azure/architecture/"

    return [
        TextBlock.create(
            document_id=document_id,
            block_type="heading",
            text="Azure reliability",
            block_index=0,
            section_id="reliability",
            section_heading="Azure reliability",
            heading_path=[
                "Azure architecture",
                "Reliability",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
        TextBlock.create(
            document_id=document_id,
            block_type="paragraph",
            text=(
                "Azure availability zones improve application reliability. "
                "They provide physically separate datacenter locations "
                "within an Azure region. "
                "Redundancy helps workloads tolerate infrastructure failures. "
                "Monitoring allows teams to detect service degradation."
            ),
            block_index=1,
            section_id="reliability",
            section_heading="Azure reliability",
            heading_path=[
                "Azure architecture",
                "Reliability",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
        TextBlock.create(
            document_id=document_id,
            block_type="heading",
            text="Azure cost management",
            block_index=2,
            section_id="cost-management",
            section_heading="Azure cost management",
            heading_path=[
                "Azure architecture",
                "Cost management",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
        TextBlock.create(
            document_id=document_id,
            block_type="paragraph",
            text=(
                "Azure Cost Management provides billing reports. "
                "Budgets help organizations control cloud expenses. "
                "Cost alerts notify teams when spending exceeds a threshold. "
                "Resource tagging improves financial reporting."
            ),
            block_index=3,
            section_id="cost-management",
            section_heading="Azure cost management",
            heading_path=[
                "Azure architecture",
                "Cost management",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
        TextBlock.create(
            document_id=document_id,
            block_type="heading",
            text="Azure identity security",
            block_index=4,
            section_id="identity-security",
            section_heading="Azure identity security",
            heading_path=[
                "Azure architecture",
                "Security",
                "Identity",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
        TextBlock.create(
            document_id=document_id,
            block_type="paragraph",
            text=(
                "Microsoft Entra ID manages user identities and access. "
                "Multifactor authentication reduces account compromise risks. "
                "Role-based access control limits permissions. "
                "Security administrators should regularly review access rights."
            ),
            block_index=5,
            section_id="identity-security",
            section_heading="Azure identity security",
            heading_path=[
                "Azure architecture",
                "Security",
                "Identity",
            ],
            source_url=source_url,
            document_title=document_title,
            language="en",
        ),
    ]


def create_encoder() -> SentenceTransformerEncoder:
    """
    Crée l'encodeur réel SentenceTransformer.
    """

    return SentenceTransformerEncoder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=8,
        normalize_embeddings=True,
    )


def create_chunker(
    *,
    chunk_size: int = 40,
    chunk_overlap: int = 4,
    threshold: float = 0.45,
) -> SemanticChunker:
    """
    Crée le SemanticChunker avec le vrai modèle d'embedding.
    """

    return SemanticChunker(
        config=create_config(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            threshold=threshold,
        ),
        encoder=create_encoder(),
        tokenizer=WhitespaceTokenizer(),
    )


def display_chunks(
    chunks: Sequence[DocumentChunk],
) -> None:
    """
    Affiche les attributs réellement disponibles dans DocumentChunk.
    """

    print("Nombre de chunks :", len(chunks))
    print()

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index} "
            f"| tokens={chunk.token_count} "
            f"| caractères={chunk.character_count}"
        )

        print(
            "Chunk ID :",
            chunk.chunk_id,
        )

        print(
            "Document ID :",
            chunk.document_id,
        )

        print(
            "Stratégie :",
            chunk.strategy.value,
        )

        print(
            "Experiment ID :",
            chunk.experiment_id,
        )

        print(
            "Section ID :",
            chunk.section_id,
        )

        print(
            "Titre de section :",
            chunk.section_heading,
        )

        print(
            "Heading path :",
            chunk.heading_path,
        )

        print(
            "Types de contenu :",
            chunk.content_types,
        )

        print(
            "URL source :",
            chunk.source_url,
        )

        print(
            "Titre du document :",
            chunk.document_title,
        )

        print(
            "Raison de frontière :",
            chunk.metadata.get(
                "boundary_reason",
                "non renseignée",
            ),
        )

        print(
            "Similarité minimale :",
            chunk.metadata.get(
                "minimum_similarity",
                "non renseignée",
            ),
        )

        print(
            "Overlap déclaré :",
            chunk.metadata.get(
                "overlap_token_count",
                0,
            ),
        )

        print(
            "Encodeur :",
            chunk.metadata.get(
                "semantic_encoder",
                "non renseigné",
            ),
        )

        print(
            "Seuil sémantique :",
            chunk.metadata.get(
                "semantic_similarity_threshold",
                "non renseigné",
            ),
        )

        print(
            "Texte :",
            chunk.text,
        )

        print("-" * 70)


def test_real_semantic_chunking() -> list[DocumentChunk]:
    """
    Vérifie le fonctionnement général du SemanticChunker réel.
    """

    separator()
    print("TEST DU SEMANTIC CHUNKER AVEC LE VRAI MODELE")
    separator()

    chunker = create_chunker()

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    display_chunks(chunks)

    assert chunks, (
        "Le SemanticChunker n'a produit aucun chunk."
    )

    for expected_index, chunk in enumerate(chunks):
        assert isinstance(
            chunk,
            DocumentChunk,
        )

        assert chunk.chunk_index == expected_index

        assert chunk.chunk_id.startswith(
            "chunk_"
        )

        assert chunk.document_id == (
            "azure-semantic-real-test"
        )

        assert chunk.strategy == (
            ChunkingStrategy.SEMANTIC
        )

        assert chunk.chunk_size == 40

        assert chunk.chunk_overlap == 4

        assert chunk.text.strip()

        assert chunk.token_count > 0

        assert chunk.token_count <= 40

        assert chunk.character_count == len(
            chunk.text
        )

        assert chunk.experiment_id == (
            "semantic_40_4"
        )

    print(
        "Découpage sémantique réel : OK"
    )

    return chunks


def test_topics_are_preserved(
    chunks: Sequence[DocumentChunk],
) -> None:
    """
    Vérifie que les trois thèmes du document sont conservés.
    """

    print()
    separator()
    print("TEST DE PRESERVATION DES SUJETS")
    separator()

    complete_text = " ".join(
        chunk.text.lower()
        for chunk in chunks
    )

    reliability_present = any(
        expression in complete_text
        for expression in (
            "reliability",
            "availability zones",
            "redundancy",
        )
    )

    cost_present = any(
        expression in complete_text
        for expression in (
            "cost management",
            "billing",
            "budgets",
        )
    )

    security_present = any(
        expression in complete_text
        for expression in (
            "identity security",
            "entra id",
            "multifactor authentication",
        )
    )

    assert reliability_present, (
        "Le sujet de la fiabilité a été perdu."
    )

    assert cost_present, (
        "Le sujet des coûts a été perdu."
    )

    assert security_present, (
        "Le sujet de la sécurité a été perdu."
    )

    print("Sujet fiabilité : présent")
    print("Sujet coûts : présent")
    print("Sujet sécurité : présent")
    print("Préservation des sujets : OK")


def test_size_limit() -> None:
    """
    Vérifie que chunk_size est toujours respecté.
    """

    print()
    separator()
    print("TEST DE LA TAILLE MAXIMALE")
    separator()

    maximum_size = 20

    chunker = create_chunker(
        chunk_size=maximum_size,
        chunk_overlap=3,
        threshold=0.45,
    )

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    assert chunks

    for chunk in chunks:
        print(
            f"Chunk {chunk.chunk_index} : "
            f"{chunk.token_count} tokens"
        )

        assert chunk.token_count <= maximum_size

        assert chunk.chunk_size == maximum_size

        assert chunk.chunk_overlap == 3

    print(
        "Respect de la taille maximale : OK"
    )


def test_overlap() -> None:
    """
    Vérifie l'overlap déclaré et l'overlap réel.
    """

    print()
    separator()
    print("TEST DE L'OVERLAP REEL")
    separator()

    maximum_overlap = 3
    tokenizer = WhitespaceTokenizer()

    chunker = create_chunker(
        chunk_size=20,
        chunk_overlap=maximum_overlap,
        threshold=0.45,
    )

    chunks = chunker.chunk_blocks(
        create_blocks()
    )

    assert chunks

    overlap_detected = False

    for index, chunk in enumerate(chunks):
        overlap = int(
            chunk.metadata.get(
                "overlap_token_count",
                0,
            )
        )

        print(
            f"Chunk {index} : "
            f"overlap_token_count={overlap}"
        )

        assert 0 <= overlap <= maximum_overlap

        if index == 0:
            assert overlap == 0
            continue

        if overlap == 0:
            continue

        overlap_detected = True

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

        assert previous_suffix == current_prefix

    assert overlap_detected, (
        "Aucun overlap réel n'a été détecté."
    )

    print(
        "Overlap réel : OK"
    )


def test_deterministic_ids() -> None:
    """
    Vérifie que deux exécutions identiques produisent
    les mêmes identifiants.
    """

    print()
    separator()
    print("TEST DES IDENTIFIANTS DETERMINISTES")
    separator()

    chunker = create_chunker()

    first_chunks = chunker.chunk_blocks(
        create_blocks()
    )

    second_chunks = chunker.chunk_blocks(
        create_blocks()
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first_chunks
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_chunks
    ]

    print(
        "Premier découpage :",
        first_ids,
    )

    print(
        "Deuxième découpage :",
        second_ids,
    )

    assert first_ids == second_ids

    print(
        "Identifiants déterministes : OK"
    )


def test_serialization(
    chunks: Sequence[DocumentChunk],
) -> None:
    """
    Vérifie que les chunks peuvent être convertis en dictionnaires.
    """

    print()
    separator()
    print("TEST DE SERIALISATION")
    separator()

    for chunk in chunks:
        chunk_data = chunk.to_dict()

        assert isinstance(
            chunk_data,
            dict,
        )

        assert chunk_data["chunk_id"] == (
            chunk.chunk_id
        )

        assert chunk_data["strategy"] == (
            "semantic"
        )

        assert chunk_data["experiment_id"] == (
            "semantic_40_4"
        )

        assert chunk_data["section_id"] == (
            chunk.section_id
        )

        assert chunk_data["heading_path"] == (
            chunk.heading_path
        )

    print(
        "Sérialisation des chunks : OK"
    )


def main() -> None:
    chunks = test_real_semantic_chunking()

    test_topics_are_preserved(
        chunks
    )

    test_size_limit()

    test_overlap()

    test_deterministic_ids()

    test_serialization(
        chunks
    )

    print()
    separator()
    print(
        "SEMANTIC CHUNKER AVEC MODELE REEL FONCTIONNEL."
    )
    separator()


if __name__ == "__main__":
    main()