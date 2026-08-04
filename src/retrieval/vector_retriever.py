from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.storage.pgvector_repository import database_connection


EXPECTED_EMBEDDING_DIMENSION = 768
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class VectorSearchResult:
    """
    Représente un chunk retourné par pgvector.
    """

    rank: int
    chunk_position: int
    chunk_id: str
    content: str
    source: str
    document_format: str
    page_number: int | None
    metadata: dict[str, Any]
    cosine_distance: float
    cosine_similarity: float


def validate_query_embedding(
    query_embedding: np.ndarray,
) -> np.ndarray:
    """
    Vérifie et prépare le vecteur de requête.

    Le vecteur attendu possède :
    - une dimension de 768 ;
    - aucune valeur NaN ou infinie ;
    - une norme L2 non nulle.
    """

    vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if vector.shape != (
        EXPECTED_EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "Dimension incorrecte pour la requête : "
            f"{vector.shape}. "
            f"Dimension attendue : "
            f"({EXPECTED_EMBEDDING_DIMENSION},)."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Le vecteur de requête contient des valeurs "
            "NaN ou infinies."
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm == 0.0:
        raise ValueError(
            "Le vecteur de requête possède une norme nulle."
        )

    # Sécurité : normaliser le vecteur pour utiliser
    # correctement la similarité cosinus.
    vector = vector / norm

    return vector.astype(
        np.float32,
        copy=False,
    )


class PgVectorRetriever:
    """
    Effectue une recherche vectorielle dans PostgreSQL.

    La colonne embedding est interrogée avec l'opérateur
    de distance cosinus pgvector : <=>
    """

    def __init__(
        self,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k doit être supérieur à zéro."
            )

        self.default_top_k = default_top_k

    def count_indexed_chunks(self) -> int:
        """
        Retourne le nombre de chunks présents dans PostgreSQL.
        """

        with database_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM document_chunks;
                    """
                )

                result = cursor.fetchone()

        return int(result[0])

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        document_format: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        Recherche les chunks les plus proches d'un embedding.

        Args:
            query_embedding:
                Vecteur E5 normalisé ou non, de dimension 768.

            top_k:
                Nombre maximal de résultats à retourner.

            document_format:
                Filtre optionnel : html, markdown ou pdf.

        Returns:
            Une liste ordonnée de VectorSearchResult.
        """

        vector = validate_query_embedding(
            query_embedding
        )

        limit = (
            self.default_top_k
            if top_k is None
            else int(top_k)
        )

        if limit <= 0:
            raise ValueError(
                "top_k doit être supérieur à zéro."
            )

        parameters: list[Any] = [
            vector,
        ]

        format_filter = ""

        if document_format is not None:
            normalized_format = (
                str(document_format)
                .strip()
                .lower()
            )

            if normalized_format not in {
                "html",
                "markdown",
                "pdf",
            }:
                raise ValueError(
                    "document_format doit être "
                    "html, markdown ou pdf."
                )

            format_filter = (
                "WHERE document_format = %s"
            )

            parameters.append(
                normalized_format
            )

        # Le vecteur est ajouté une deuxième fois :
        # une fois pour calculer la distance affichée,
        # une fois pour ordonner les résultats.
        parameters.extend(
            [
                vector,
                limit,
            ]
        )

        sql_query = f"""
            SELECT
                chunk_position,
                chunk_id,
                content,
                source,
                document_format,
                page_number,
                metadata,
                embedding <=> %s AS cosine_distance
            FROM document_chunks
            {format_filter}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """

        with database_connection() as connection:
            with connection.cursor() as cursor:

                cursor.execute(
                    sql_query,
                    parameters,
                )

                database_rows = cursor.fetchall()

        results: list[VectorSearchResult] = []

        for rank, row in enumerate(
            database_rows,
            start=1,
        ):
            cosine_distance = float(
                row[7]
            )

            cosine_similarity = (
                1.0 - cosine_distance
            )

            results.append(
                VectorSearchResult(
                    rank=rank,
                    chunk_position=int(row[0]),
                    chunk_id=str(row[1]),
                    content=str(row[2]),
                    source=str(row[3]),
                    document_format=str(row[4]),
                    page_number=(
                        int(row[5])
                        if row[5] is not None
                        else None
                    ),
                    metadata=dict(
                        row[6] or {}
                    ),
                    cosine_distance=cosine_distance,
                    cosine_similarity=cosine_similarity,
                )
            )

        return results