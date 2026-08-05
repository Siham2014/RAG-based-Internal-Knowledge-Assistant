from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT
    / "config"
    / "settings.yaml"
)


# ============================================================
# Modèles de configuration
# ============================================================

@dataclass(frozen=True)
class ProjectSettings:
    name: str
    environment: str


@dataclass(frozen=True)
class CorpusSettings:
    company: str
    processed_directory: Path
    chunks_path: Path


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str
    chunk_size: int
    overlap: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str
    dimension: int
    normalize: bool
    query_prefix: str
    document_prefix: str
    batch_size: int
    device: str


@dataclass(frozen=True)
class RetrievalSettings:
    vector_backend: str
    lexical_backend: str
    candidate_k: int
    hybrid_top_k: int
    final_top_k: int
    rrf_constant: int


@dataclass(frozen=True)
class RerankingSettings:
    enabled: bool
    model_name: str
    batch_size: int
    max_length: int
    device: str


@dataclass(frozen=True)
class ConfidenceSettings:
    enabled: bool
    threshold: float | None
    minimum_top1_score: float | None
    minimum_margin: float | None


@dataclass(frozen=True)
class GenerationSettings:
    enabled: bool
    provider: str
    model_name: str | None
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class StorageSettings:
    bm25_index_path: Path


@dataclass(frozen=True)
class APISettings:
    host: str
    port: int


@dataclass(frozen=True)
class ApplicationSettings:
    project: ProjectSettings
    corpus: CorpusSettings
    chunking: ChunkingSettings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    reranking: RerankingSettings
    confidence: ConfidenceSettings
    generation: GenerationSettings
    storage: StorageSettings
    api: APISettings

    source_path: Path


# ============================================================
# Fonctions de validation
# ============================================================

def _require_mapping(
    value: Any,
    section_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"La section '{section_name}' doit être un objet YAML."
        )

    return value


def _require_text(
    value: Any,
    field_name: str,
) -> str:
    normalized = str(
        value if value is not None else ""
    ).strip()

    if not normalized:
        raise ValueError(
            f"Le champ '{field_name}' ne peut pas être vide."
        )

    return normalized


def _require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Le champ '{field_name}' doit être un entier."
        ) from error

    if normalized <= 0:
        raise ValueError(
            f"Le champ '{field_name}' doit être supérieur à zéro."
        )

    return normalized


def _require_non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Le champ '{field_name}' doit être un entier."
        ) from error

    if normalized < 0:
        raise ValueError(
            f"Le champ '{field_name}' doit être positif ou nul."
        )

    return normalized


def _require_boolean(
    value: Any,
    field_name: str,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "on",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "off",
        }:
            return False

    raise ValueError(
        f"Le champ '{field_name}' doit être un booléen."
    )


def _optional_float(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Le champ '{field_name}' doit être "
            "un nombre ou null."
        ) from error


def _resolve_project_path(
    value: Any,
    field_name: str,
) -> Path:
    raw_path = _require_text(
        value,
        field_name,
    )

    path = Path(raw_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def _apply_environment_overrides(
    raw: dict[str, Any],
) -> dict[str, Any]:
    """
    Permet de modifier quelques paramètres sans éditer le YAML.

    Exemples :
        RAG_COMPANY=company_b
        RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
        RAG_DEVICE=cpu
    """

    corpus = raw.setdefault(
        "corpus",
        {},
    )

    embedding = raw.setdefault(
        "embedding",
        {},
    )

    reranking = raw.setdefault(
        "reranking",
        {},
    )

    generation = raw.setdefault(
        "generation",
        {},
    )

    overrides = {
        "RAG_COMPANY": (
            corpus,
            "company",
        ),
        "RAG_EMBEDDING_MODEL": (
            embedding,
            "model_name",
        ),
        "RAG_RERANKER_MODEL": (
            reranking,
            "model_name",
        ),
        "RAG_DEVICE": (
            embedding,
            "device",
        ),
        "RAG_GENERATION_PROVIDER": (
            generation,
            "provider",
        ),
        "RAG_GENERATION_MODEL": (
            generation,
            "model_name",
        ),
    }

    for environment_name, (
        section,
        field_name,
    ) in overrides.items():

        environment_value = os.getenv(
            environment_name
        )

        if environment_value is not None:
            section[field_name] = (
                environment_value
            )

    return raw


# ============================================================
# Chargement principal
# ============================================================

def load_settings(
    settings_path: Path | str | None = None,
) -> ApplicationSettings:
    """
    Charge, valide et retourne la configuration complète.

    Le chemin peut être défini de trois façons :

    1. argument settings_path ;
    2. variable RAG_SETTINGS_PATH ;
    3. config/settings.yaml par défaut.
    """

    environment_path = os.getenv(
        "RAG_SETTINGS_PATH"
    )

    selected_path = (
        Path(settings_path)
        if settings_path is not None
        else (
            Path(environment_path)
            if environment_path
            else DEFAULT_SETTINGS_PATH
        )
    )

    if not selected_path.is_absolute():
        selected_path = (
            PROJECT_ROOT
            / selected_path
        )

    selected_path = selected_path.resolve()

    if not selected_path.is_file():
        raise FileNotFoundError(
            "Fichier de configuration introuvable : "
            f"{selected_path}"
        )

    with selected_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if raw is None:
        raise ValueError(
            "Le fichier de configuration est vide."
        )

    raw = _require_mapping(
        raw,
        "racine",
    )

    raw = _apply_environment_overrides(
        raw
    )

    project_raw = _require_mapping(
        raw.get("project"),
        "project",
    )

    corpus_raw = _require_mapping(
        raw.get("corpus"),
        "corpus",
    )

    chunking_raw = _require_mapping(
        raw.get("chunking"),
        "chunking",
    )

    embedding_raw = _require_mapping(
        raw.get("embedding"),
        "embedding",
    )

    retrieval_raw = _require_mapping(
        raw.get("retrieval"),
        "retrieval",
    )

    reranking_raw = _require_mapping(
        raw.get("reranking"),
        "reranking",
    )

    confidence_raw = _require_mapping(
        raw.get("confidence"),
        "confidence",
    )

    generation_raw = _require_mapping(
        raw.get("generation"),
        "generation",
    )

    storage_raw = _require_mapping(
        raw.get("storage"),
        "storage",
    )

    api_raw = _require_mapping(
        raw.get("api"),
        "api",
    )

    settings = ApplicationSettings(
        project=ProjectSettings(
            name=_require_text(
                project_raw.get("name"),
                "project.name",
            ),
            environment=_require_text(
                project_raw.get(
                    "environment"
                ),
                "project.environment",
            ),
        ),

        corpus=CorpusSettings(
            company=_require_text(
                corpus_raw.get("company"),
                "corpus.company",
            ),
            processed_directory=(
                _resolve_project_path(
                    corpus_raw.get(
                        "processed_directory"
                    ),
                    "corpus.processed_directory",
                )
            ),
            chunks_path=_resolve_project_path(
                corpus_raw.get(
                    "chunks_path"
                ),
                "corpus.chunks_path",
            ),
        ),

        chunking=ChunkingSettings(
            strategy=_require_text(
                chunking_raw.get("strategy"),
                "chunking.strategy",
            ),
            chunk_size=(
                _require_positive_integer(
                    chunking_raw.get(
                        "chunk_size"
                    ),
                    "chunking.chunk_size",
                )
            ),
            overlap=(
                _require_non_negative_integer(
                    chunking_raw.get(
                        "overlap"
                    ),
                    "chunking.overlap",
                )
            ),
        ),

        embedding=EmbeddingSettings(
            model_name=_require_text(
                embedding_raw.get(
                    "model_name"
                ),
                "embedding.model_name",
            ),
            dimension=(
                _require_positive_integer(
                    embedding_raw.get(
                        "dimension"
                    ),
                    "embedding.dimension",
                )
            ),
            normalize=_require_boolean(
                embedding_raw.get(
                    "normalize"
                ),
                "embedding.normalize",
            ),
            query_prefix=str(
                embedding_raw.get(
                    "query_prefix",
                    "",
                )
            ),
            document_prefix=str(
                embedding_raw.get(
                    "document_prefix",
                    "",
                )
            ),
            batch_size=(
                _require_positive_integer(
                    embedding_raw.get(
                        "batch_size"
                    ),
                    "embedding.batch_size",
                )
            ),
            device=_require_text(
                embedding_raw.get("device"),
                "embedding.device",
            ),
        ),

        retrieval=RetrievalSettings(
            vector_backend=_require_text(
                retrieval_raw.get(
                    "vector_backend"
                ),
                "retrieval.vector_backend",
            ),
            lexical_backend=_require_text(
                retrieval_raw.get(
                    "lexical_backend"
                ),
                "retrieval.lexical_backend",
            ),
            candidate_k=(
                _require_positive_integer(
                    retrieval_raw.get(
                        "candidate_k"
                    ),
                    "retrieval.candidate_k",
                )
            ),
            hybrid_top_k=(
                _require_positive_integer(
                    retrieval_raw.get(
                        "hybrid_top_k"
                    ),
                    "retrieval.hybrid_top_k",
                )
            ),
            final_top_k=(
                _require_positive_integer(
                    retrieval_raw.get(
                        "final_top_k"
                    ),
                    "retrieval.final_top_k",
                )
            ),
            rrf_constant=(
                _require_non_negative_integer(
                    retrieval_raw.get(
                        "rrf_constant"
                    ),
                    "retrieval.rrf_constant",
                )
            ),
        ),

        reranking=RerankingSettings(
            enabled=_require_boolean(
                reranking_raw.get("enabled"),
                "reranking.enabled",
            ),
            model_name=_require_text(
                reranking_raw.get(
                    "model_name"
                ),
                "reranking.model_name",
            ),
            batch_size=(
                _require_positive_integer(
                    reranking_raw.get(
                        "batch_size"
                    ),
                    "reranking.batch_size",
                )
            ),
            max_length=(
                _require_positive_integer(
                    reranking_raw.get(
                        "max_length"
                    ),
                    "reranking.max_length",
                )
            ),
            device=_require_text(
                reranking_raw.get("device"),
                "reranking.device",
            ),
        ),

        confidence=ConfidenceSettings(
            enabled=_require_boolean(
                confidence_raw.get(
                    "enabled"
                ),
                "confidence.enabled",
            ),
            threshold=_optional_float(
                confidence_raw.get(
                    "threshold"
                ),
                "confidence.threshold",
            ),
            minimum_top1_score=(
                _optional_float(
                    confidence_raw.get(
                        "minimum_top1_score"
                    ),
                    (
                        "confidence."
                        "minimum_top1_score"
                    ),
                )
            ),
            minimum_margin=_optional_float(
                confidence_raw.get(
                    "minimum_margin"
                ),
                "confidence.minimum_margin",
            ),
        ),

        generation=GenerationSettings(
            enabled=_require_boolean(
                generation_raw.get(
                    "enabled"
                ),
                "generation.enabled",
            ),
            provider=_require_text(
                generation_raw.get(
                    "provider"
                ),
                "generation.provider",
            ),
            model_name=(
                str(
                    generation_raw[
                        "model_name"
                    ]
                ).strip()
                if generation_raw.get(
                    "model_name"
                ) is not None
                else None
            ),
            temperature=float(
                generation_raw.get(
                    "temperature",
                    0.0,
                )
            ),
            max_tokens=(
                _require_positive_integer(
                    generation_raw.get(
                        "max_tokens"
                    ),
                    "generation.max_tokens",
                )
            ),
        ),

        storage=StorageSettings(
            bm25_index_path=(
                _resolve_project_path(
                    storage_raw.get(
                        "bm25_index_path"
                    ),
                    (
                        "storage."
                        "bm25_index_path"
                    ),
                )
            ),
        ),

        api=APISettings(
            host=_require_text(
                api_raw.get("host"),
                "api.host",
            ),
            port=_require_positive_integer(
                api_raw.get("port"),
                "api.port",
            ),
        ),

        source_path=selected_path,
    )

    _validate_cross_section_rules(
        settings
    )

    return settings


def _validate_cross_section_rules(
    settings: ApplicationSettings,
) -> None:
    """
    Vérifie les relations entre plusieurs paramètres.
    """

    if (
        settings.chunking.overlap
        >= settings.chunking.chunk_size
    ):
        raise ValueError(
            "chunking.overlap doit être inférieur "
            "à chunking.chunk_size."
        )

    if (
        settings.retrieval.hybrid_top_k
        > settings.retrieval.candidate_k * 2
    ):
        raise ValueError(
            "retrieval.hybrid_top_k est trop élevé "
            "par rapport à retrieval.candidate_k."
        )

    if (
        settings.retrieval.final_top_k
        > settings.retrieval.hybrid_top_k
    ):
        raise ValueError(
            "retrieval.final_top_k ne peut pas "
            "dépasser retrieval.hybrid_top_k."
        )

    if settings.embedding.device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "embedding.device doit être "
            "auto, cpu ou cuda."
        )

    if settings.reranking.device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "reranking.device doit être "
            "auto, cpu ou cuda."
        )

    if settings.embedding.dimension != 768:
        raise ValueError(
            "La base pgvector actuelle attend une "
            "dimension d'embedding égale à 768."
        )

    if not 0.0 <= (
        settings.generation.temperature
    ) <= 2.0:
        raise ValueError(
            "generation.temperature doit être "
            "comprise entre 0 et 2."
        )


_SETTINGS_CACHE: ApplicationSettings | None = None


def get_settings(
    force_reload: bool = False,
) -> ApplicationSettings:
    """
    Retourne une instance mise en cache.

    Dans FastAPI, la configuration sera ainsi chargée
    une seule fois au démarrage.
    """

    global _SETTINGS_CACHE

    if (
        _SETTINGS_CACHE is None
        or force_reload
    ):
        _SETTINGS_CACHE = load_settings()

    return _SETTINGS_CACHE