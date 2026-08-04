from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb

from src.storage.pgvector_repository import database_connection


# ============================================================
# Chemins du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "fixed_1024_chunks.jsonl"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "e5_base_document_embeddings.npy"
)

EXPECTED_NUMBER_OF_CHUNKS = 1478
EXPECTED_EMBEDDING_DIMENSION = 768

BATCH_SIZE = 200


# ============================================================
# Fonctions utilitaires
# ============================================================

def require_file(path: Path) -> None:
    """
    Vérifie qu’un fichier nécessaire existe.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def canonical_source(source: str) -> str:
    """
    Normalise les séparateurs Windows et Linux.
    """

    return str(source).replace("\\", "/")


def detect_document_format(source: str) -> str:
    """
    Déduit le format du document depuis son extension.
    """

    source_lower = str(source).lower()

    if source_lower.endswith(".md"):
        return "markdown"

    if source_lower.endswith(".html"):
        return "html"

    if source_lower.endswith(".pdf"):
        return "pdf"

    return "unknown"


def safe_integer(value: Any) -> int | None:
    """
    Convertit une valeur en entier si possible.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Chargement des données
# ============================================================

def load_chunks(path: Path) -> list[dict[str, Any]]:
    """
    Charge les chunks depuis le fichier JSONL.
    """

    chunks: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide à la ligne {line_number}."
                ) from error

            content = str(
                record.get("page_content", "")
            ).strip()

            if not content:
                raise ValueError(
                    f"Chunk vide à la ligne {line_number}."
                )

            metadata = record.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Métadonnées invalides à la ligne "
                    f"{line_number}."
                )

            chunks.append(record)

    return chunks


def load_embeddings(path: Path) -> np.ndarray:
    """
    Charge la matrice NumPy des embeddings E5.
    """

    embeddings = np.load(
        path,
        allow_pickle=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    return embeddings


# ============================================================
# Validation
# ============================================================

def validate_assets(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """
    Vérifie l’alignement entre les chunks
    et les embeddings.
    """

    if len(chunks) != EXPECTED_NUMBER_OF_CHUNKS:
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_CHUNKS} chunks attendus, "
            f"{len(chunks)} trouvés."
        )

    expected_shape = (
        EXPECTED_NUMBER_OF_CHUNKS,
        EXPECTED_EMBEDDING_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Forme attendue : {expected_shape}, "
            f"forme trouvée : {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs "
            "NaN ou infinies."
        )

    chunk_ids: list[str] = []

    for position, record in enumerate(chunks):

        metadata = record.get(
            "metadata",
            {},
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
                record.get(
                    "chunk_id",
                    f"chunk_{position}",
                ),
            )
        ).strip()

        if not chunk_id:
            raise ValueError(
                f"Chunk ID vide à la position {position}."
            )

        chunk_ids.append(chunk_id)

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(
            "Des chunk_id dupliqués ont été détectés."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print("Validation réussie")
    print("Chunks :", len(chunks))
    print("Embeddings :", embeddings.shape)
    print(
        "Norme moyenne :",
        round(float(norms.mean()), 6),
    )


# ============================================================
# Préparation des lignes PostgreSQL
# ============================================================

def build_database_rows(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> list[tuple]:
    """
    Transforme les chunks et embeddings en lignes
    compatibles avec PostgreSQL.
    """

    rows: list[tuple] = []

    for position, (
        record,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        metadata = dict(
            record.get("metadata", {})
        )

        content = str(
            record["page_content"]
        ).strip()

        chunk_id = str(
            metadata.get(
                "chunk_id",
                record.get(
                    "chunk_id",
                    f"chunk_{position}",
                ),
            )
        ).strip()

        source = canonical_source(
            metadata.get(
                "source",
                record.get(
                    "source",
                    "unknown",
                ),
            )
        )

        document_format = str(
            metadata.get(
                "format",
                metadata.get(
                    "document_format",
                    detect_document_format(source),
                ),
            )
        ).lower()

        page_number = safe_integer(
            metadata.get(
                "page",
                metadata.get(
                    "page_number",
                ),
            )
        )

        # Conserver la position pour assurer la traçabilité
        metadata["chunk_position"] = position

        rows.append(
            (
                position,
                chunk_id,
                content,
                source,
                document_format,
                page_number,
                Jsonb(metadata),
                embedding,
            )
        )

    return rows


# ============================================================
# Ingestion PostgreSQL
# ============================================================

INSERT_SQL = """
    INSERT INTO document_chunks (
        chunk_position,
        chunk_id,
        content,
        source,
        document_format,
        page_number,
        metadata,
        embedding
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s
    )
    ON CONFLICT (chunk_id)
    DO UPDATE SET
        chunk_position = EXCLUDED.chunk_position,
        content = EXCLUDED.content,
        source = EXCLUDED.source,
        document_format = EXCLUDED.document_format,
        page_number = EXCLUDED.page_number,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding;
"""


def clear_table() -> None:
    """
    Vide la table avant une ingestion complète.
    """

    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "TRUNCATE TABLE document_chunks "
                "RESTART IDENTITY;"
            )

    print("Table document_chunks vidée.")


def insert_rows(
    rows: list[tuple],
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Insère les données par lots.
    """

    total_rows = len(rows)

    with database_connection() as connection:

        with connection.cursor() as cursor:

            for start_index in range(
                0,
                total_rows,
                batch_size,
            ):

                batch = rows[
                    start_index:
                    start_index + batch_size
                ]

                cursor.executemany(
                    INSERT_SQL,
                    batch,
                )

                inserted_until = min(
                    start_index + len(batch),
                    total_rows,
                )

                print(
                    f"Insertion : "
                    f"{inserted_until}/{total_rows}"
                )


def count_database_rows() -> int:
    """
    Compte les chunks présents dans PostgreSQL.
    """

    with database_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT COUNT(*) "
                "FROM document_chunks;"
            )

            result = cursor.fetchone()

    return int(result[0])


def show_database_distribution() -> None:
    """
    Affiche la répartition des chunks par format.
    """

    with database_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    document_format,
                    COUNT(*)
                FROM document_chunks
                GROUP BY document_format
                ORDER BY document_format;
                """
            )

            rows = cursor.fetchall()

    print("\nRépartition PostgreSQL :")

    for document_format, count in rows:
        print(
            f"- {document_format}: {count}"
        )


# ============================================================
# Programme principal
# ============================================================

def main() -> int:
    """
    Exécute l’ingestion complète.
    """

    print("=" * 80)
    print("INGESTION DES CHUNKS DANS PGVECTOR")
    print("=" * 80)

    require_file(CHUNKS_PATH)
    require_file(EMBEDDINGS_PATH)

    print("Chunks :", CHUNKS_PATH)
    print("Embeddings :", EMBEDDINGS_PATH)

    start_time = time.perf_counter()

    chunks = load_chunks(
        CHUNKS_PATH
    )

    embeddings = load_embeddings(
        EMBEDDINGS_PATH
    )

    validate_assets(
        chunks=chunks,
        embeddings=embeddings,
    )

    database_rows = build_database_rows(
        chunks=chunks,
        embeddings=embeddings,
    )

    clear_table()

    insert_rows(
        rows=database_rows,
        batch_size=BATCH_SIZE,
    )

    database_count = count_database_rows()

    if database_count != len(chunks):
        raise RuntimeError(
            f"Ingestion incomplète : "
            f"{database_count}/{len(chunks)} lignes."
        )

    show_database_distribution()

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 80)
    print("INGESTION TERMINÉE")
    print("=" * 80)

    print(
        "Lignes PostgreSQL :",
        database_count,
    )

    print(
        "Temps total :",
        round(elapsed_time, 3),
        "secondes",
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except Exception as error:
        print(
            f"\nERREUR : {error}",
            file=sys.stderr,
        )

        raise SystemExit(1)