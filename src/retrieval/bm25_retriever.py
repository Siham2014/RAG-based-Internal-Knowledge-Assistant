from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "lexical"
    / "bm25_index.db"
)

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class BM25SearchResult:
    """
    Représente un chunk retourné par SQLite FTS5.
    """

    rank: int
    chunk_position: int
    chunk_id: str
    content: str
    source: str
    document_format: str
    page_number: int | None
    metadata_json: str
    bm25_score: float


def normalize_query_text(text: str) -> str:
    """
    Normalise une question avant de l'envoyer à FTS5.
    """

    normalized = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    normalized = normalized.strip()

    if not normalized:
        raise ValueError(
            "La requête BM25 ne peut pas être vide."
        )

    return normalized


def build_fts5_query(text: str) -> str:
    """
    Transforme une question naturelle en requête FTS5 robuste.

    Exemple :
        "Who shares responsibility for sustainability?"
    devient approximativement :
        "shares OR responsibility OR sustainability"
    """

    normalized = normalize_query_text(text)

    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_-]*",
        normalized.lower(),
    )

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    filtered_tokens = [
        token
        for token in tokens
        if (
            token not in stopwords
            and len(token) >= 2
        )
    ]

    if not filtered_tokens:
        filtered_tokens = tokens

    unique_tokens: list[str] = []

    seen_tokens: set[str] = set()

    for token in filtered_tokens:
        if token not in seen_tokens:
            seen_tokens.add(token)
            unique_tokens.append(token)

    if not unique_tokens:
        raise ValueError(
            "Aucun terme lexical exploitable "
            "dans la question."
        )

    return " OR ".join(
        f'"{token}"'
        for token in unique_tokens
    )


class BM25Retriever:
    """
    Effectue une recherche lexicale avec SQLite FTS5
    et la fonction de classement bm25().
    """

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE_PATH,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:

        self.database_path = Path(
            database_path
        )

        if default_top_k <= 0:
            raise ValueError(
                "default_top_k doit être supérieur à zéro."
            )

        self.default_top_k = default_top_k

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "Index BM25 introuvable : "
                f"{self.database_path}"
            )

    def create_connection(
        self,
    ) -> sqlite3.Connection:
        """
        Ouvre une connexion SQLite.
        """

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def count_indexed_chunks(
        self,
    ) -> int:
        """
        Compte les chunks indexés.
        """

        with self.create_connection() as connection:

            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM document_chunks;
                """
            ).fetchone()

        return int(result[0])

    def search(
        self,
        question: str,
        top_k: int | None = None,
        document_format: str | None = None,
    ) -> list[BM25SearchResult]:
        """
        Recherche les chunks les plus pertinents avec BM25.
        """

        limit = (
            self.default_top_k
            if top_k is None
            else int(top_k)
        )

        if limit <= 0:
            raise ValueError(
                "top_k doit être supérieur à zéro."
            )

        fts_query = build_fts5_query(
            question
        )

        parameters: list[Any] = [
            fts_query,
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
                "AND d.document_format = ?"
            )

            parameters.append(
                normalized_format
            )

        parameters.append(
            limit
        )

        sql_query = f"""
            SELECT
                d.chunk_position,
                d.chunk_id,
                d.content,
                d.source,
                d.document_format,
                d.page_number,
                d.metadata_json,
                bm25(document_chunks_fts) AS bm25_score
            FROM document_chunks_fts
            JOIN document_chunks AS d
                ON d.id = document_chunks_fts.rowid
            WHERE document_chunks_fts MATCH ?
            {format_filter}
            ORDER BY bm25_score
            LIMIT ?;
        """

        with self.create_connection() as connection:

            rows = connection.execute(
                sql_query,
                parameters,
            ).fetchall()

        results: list[BM25SearchResult] = []

        for rank, row in enumerate(
            rows,
            start=1,
        ):

            results.append(
                BM25SearchResult(
                    rank=rank,
                    chunk_position=int(
                        row["chunk_position"]
                    ),
                    chunk_id=str(
                        row["chunk_id"]
                    ),
                    content=str(
                        row["content"]
                    ),
                    source=str(
                        row["source"]
                    ),
                    document_format=str(
                        row["document_format"]
                    ),
                    page_number=(
                        int(row["page_number"])
                        if row["page_number"]
                        is not None
                        else None
                    ),
                    metadata_json=str(
                        row["metadata_json"]
                        or "{}"
                    ),
                    bm25_score=float(
                        row["bm25_score"]
                    ),
                )
            )

        return results