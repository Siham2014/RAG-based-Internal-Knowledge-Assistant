from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


# ============================================================
# Chargement du fichier .env
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

print("=" * 70)
print("ENV PATH :", ENV_PATH)
print("DATABASE_URL :", DATABASE_URL)
print("=" * 70)


# ============================================================
# Connexion PostgreSQL
# ============================================================

def create_connection():

    connection = psycopg.connect(
        DATABASE_URL
    )

    register_vector(connection)

    return connection


@contextmanager
def database_connection():

    connection = create_connection()

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# Test
# ============================================================

def test_connection():

    with database_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT version();"
            )

            postgres_version = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT current_database();
                """
            )

            database_name = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT current_user;
                """
            )

            user_name = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT extname
                FROM pg_extension;
                """
            )

            extensions = cursor.fetchall()

            cursor.execute(
                """
                SELECT EXISTS (

                    SELECT 1

                    FROM information_schema.tables

                    WHERE table_name='document_chunks'

                );
                """
            )

            table_exists = cursor.fetchone()[0]

    print()

    print("=" * 70)
    print("Connexion réussie")
    print("=" * 70)

    print("Base :", database_name)

    print("Utilisateur :", user_name)

    print("Version PostgreSQL :")
    print(postgres_version)

    print()

    print("Extensions installées :")

    for extension in extensions:
        print("-", extension[0])

    print()

    print(
        "Table document_chunks :",
        table_exists
    )


if __name__ == "__main__":

    test_connection()