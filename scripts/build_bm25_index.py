from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "fixed_1024_chunks.jsonl"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "lexical"
    / "bm25_index.db"
)

EXPECTED_NUMBER_OF_CHUNKS = 1478


def require_file(
    path: Path,
) -> None:
    """
    Vérifie qu'un fichier existe.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


def canonical_source(
    source: str,
) -> str:
    """
    Normalise les séparateurs de chemin.
    """

    return str(source).replace(
        "\\",
        "/",
    )


def detect_document_format(
    source: str,
) -> str:
    """
    Déduit le format depuis le nom de fichier.
    """

    source_lower = source.lower()

    if source_lower.endswith(".md"):
        return "markdown"

    if source_lower.endswith(".html"):
        return "html"

    if source_lower.endswith(".pdf"):
        return "pdf"

    return "unknown"


def safe_integer(
    value: Any,
) -> int | None:
    """
    Convertit une valeur en entier si possible.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Charge le JSONL des chunks.
    """

    chunks: list[
        dict[str, Any]
    ] = []

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
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide à la ligne "
                    f"{line_number}."
                ) from error

            content = str(
                record.get(
                    "page_content",
                    "",
                )
            ).strip()

            if not content:
                raise ValueError(
                    f"Chunk vide à la ligne "
                    f"{line_number}."
                )

            chunks.append(
                record
            )

    if len(chunks) != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise ValueError(
            f"{EXPECTED_NUMBER_OF_CHUNKS} "
            f"chunks attendus, "
            f"{len(chunks)} trouvés."
        )

    return chunks


def build_rows(
    chunks: list[dict[str, Any]],
) -> list[tuple]:
    """
    Prépare les lignes SQLite.
    """

    rows: list[tuple] = []

    chunk_ids: set[str] = set()

    for position, record in enumerate(
        chunks
    ):

        metadata = dict(
            record.get(
                "metadata",
                {},
            )
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

        if not chunk_id:
            raise ValueError(
                f"chunk_id vide à la "
                f"position {position}."
            )

        if chunk_id in chunk_ids:
            raise ValueError(
                f"chunk_id dupliqué : "
                f"{chunk_id}"
            )

        chunk_ids.add(
            chunk_id
        )

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
                    detect_document_format(
                        source
                    ),
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

        metadata["chunk_position"] = (
            position
        )

        rows.append(
            (
                position,
                chunk_id,
                content,
                source,
                document_format,
                page_number,
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                ),
            )
        )

    return rows


def create_database(
    database_path: Path,
) -> sqlite3.Connection:
    """
    Crée une nouvelle base SQLite.
    """

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if database_path.exists():
        database_path.unlink()

    connection = sqlite3.connect(
        database_path
    )

    connection.execute(
        "PRAGMA journal_mode = WAL;"
    )

    connection.execute(
        "PRAGMA synchronous = NORMAL;"
    )

    connection.execute(
        """
        CREATE TABLE document_chunks (
            id INTEGER PRIMARY KEY,
            chunk_position INTEGER UNIQUE NOT NULL,
            chunk_id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            document_format TEXT NOT NULL,
            page_number INTEGER,
            metadata_json TEXT NOT NULL
        );
        """
    )

    connection.execute(
        """
        CREATE VIRTUAL TABLE document_chunks_fts
        USING fts5(
            content,
            chunk_id,
            source,
            content='document_chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_chunks_source
        ON document_chunks(source);
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_chunks_format
        ON document_chunks(document_format);
        """
    )

    return connection


def insert_rows(
    connection: sqlite3.Connection,
    rows: list[tuple],
) -> None:
    """
    Insère les chunks puis reconstruit l'index FTS5.
    """

    connection.executemany(
        """
        INSERT INTO document_chunks (
            chunk_position,
            chunk_id,
            content,
            source,
            document_format,
            page_number,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )

    connection.execute(
        """
        INSERT INTO document_chunks_fts(
            document_chunks_fts
        )
        VALUES('rebuild');
        """
    )

    connection.commit()


def verify_database(
    connection: sqlite3.Connection,
) -> None:
    """
    Vérifie la base et l'index FTS5.
    """

    chunk_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM document_chunks;
        """
    ).fetchone()[0]

    fts_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM document_chunks_fts;
        """
    ).fetchone()[0]

    if chunk_count != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise RuntimeError(
            f"Nombre incorrect dans "
            f"document_chunks : {chunk_count}."
        )

    if fts_count != (
        EXPECTED_NUMBER_OF_CHUNKS
    ):
        raise RuntimeError(
            f"Nombre incorrect dans "
            f"document_chunks_fts : "
            f"{fts_count}."
        )

    distribution = (
        connection.execute(
            """
            SELECT
                document_format,
                COUNT(*)
            FROM document_chunks
            GROUP BY document_format
            ORDER BY document_format;
            """
        ).fetchall()
    )

    print(
        "Chunks SQLite :",
        chunk_count,
    )

    print(
        "Chunks FTS5 :",
        fts_count,
    )

    print(
        "Répartition :"
    )

    for document_format, count in (
        distribution
    ):
        print(
            f"- {document_format}: {count}"
        )


def main() -> None:
    """
    Construit l'index BM25.
    """

    print("=" * 80)
    print(
        "CONSTRUCTION DE L'INDEX BM25 — SQLITE FTS5"
    )
    print("=" * 80)

    require_file(
        CHUNKS_PATH
    )

    start_time = time.perf_counter()

    chunks = load_chunks(
        CHUNKS_PATH
    )

    rows = build_rows(
        chunks
    )

    connection = create_database(
        DATABASE_PATH
    )

    try:
        insert_rows(
            connection,
            rows,
        )

        verify_database(
            connection
        )

    finally:
        connection.close()

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    database_size_mb = (
        DATABASE_PATH.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 80)
    print(
        "INDEX BM25 TERMINÉ"
    )
    print("=" * 80)

    print(
        "Base :",
        DATABASE_PATH,
    )

    print(
        "Taille :",
        round(
            database_size_mb,
            3,
        ),
        "Mo",
    )

    print(
        "Temps total :",
        round(
            elapsed_time,
            3,
        ),
        "secondes",
    )


if __name__ == "__main__":
    main()