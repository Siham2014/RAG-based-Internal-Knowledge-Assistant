from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.common.settings import (
    ApplicationSettings,
    get_settings,
)
from src.reranking import (
    CrossEncoderReranker,
    RerankedSearchResult,
)
from src.retrieval.hybrid_rrf_retriever import (
    HybridRRFRetriever,
)


# ============================================================
# Valeurs par défaut
# ============================================================

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-base-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

DEFAULT_EMBEDDING_DIMENSION = 768

DEFAULT_CANDIDATE_K = 50
DEFAULT_HYBRID_TOP_K = 20
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_CONSTANT = 60

DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_RERANKER_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512

DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_NORMALIZE_EMBEDDINGS = True


# ============================================================
# Modèles de réponse
# ============================================================

@dataclass(frozen=True)
class RetrievalTimings:
    """
    Temps d'exécution des différentes étapes du pipeline.
    """

    embedding_time_ms: float
    hybrid_retrieval_time_ms: float
    reranking_time_ms: float
    total_time_ms: float


@dataclass(frozen=True)
class RetrievalPipelineResponse:
    """
    Résultat complet du pipeline de recherche documentaire.
    """

    question: str
    results: list[RerankedSearchResult]
    timings: RetrievalTimings

    embedding_model: str
    reranker_model: str
    embedding_dimension: int

    candidate_k: int
    hybrid_top_k: int
    final_top_k: int
    rrf_constant: int

    embedding_device: str
    reranker_device: str

    document_format_filter: str | None

    def to_dict(self) -> dict[str, Any]:
        """
        Convertit la réponse en dictionnaire sérialisable en JSON.
        """

        return {
            "question": self.question,
            "configuration": {
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model,
                "embedding_dimension": (
                    self.embedding_dimension
                ),
                "candidate_k": self.candidate_k,
                "hybrid_top_k": self.hybrid_top_k,
                "final_top_k": self.final_top_k,
                "rrf_constant": self.rrf_constant,
                "embedding_device": (
                    self.embedding_device
                ),
                "reranker_device": (
                    self.reranker_device
                ),
                "document_format_filter": (
                    self.document_format_filter
                ),
            },
            "timings": {
                "embedding_time_ms": (
                    self.timings.embedding_time_ms
                ),
                "hybrid_retrieval_time_ms": (
                    self.timings.hybrid_retrieval_time_ms
                ),
                "reranking_time_ms": (
                    self.timings.reranking_time_ms
                ),
                "total_time_ms": (
                    self.timings.total_time_ms
                ),
            },
            "results": [
                {
                    "rank": result.rank,
                    "hybrid_rank": (
                        result.hybrid_rank
                    ),
                    "chunk_position": (
                        result.chunk_position
                    ),
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "source": result.source,
                    "document_format": (
                        result.document_format
                    ),
                    "page_number": (
                        result.page_number
                    ),
                    "metadata": result.metadata,
                    "reranker_score": (
                        result.reranker_score
                    ),
                    "rrf_score": (
                        result.rrf_score
                    ),
                    "vector_rank": (
                        result.vector_rank
                    ),
                    "bm25_rank": (
                        result.bm25_rank
                    ),
                    "cosine_similarity": (
                        result.cosine_similarity
                    ),
                    "cosine_distance": (
                        result.cosine_distance
                    ),
                    "bm25_score": (
                        result.bm25_score
                    ),
                }
                for result in self.results
            ],
        }


# ============================================================
# Pipeline principal
# ============================================================

class RetrievalPipeline:
    """
    Pipeline final de recherche documentaire du système RAG.

    Étapes :

    1. Encodage de la question avec le modèle d'embedding.
    2. Recherche vectorielle avec pgvector.
    3. Recherche lexicale avec BM25.
    4. Fusion des classements avec RRF.
    5. Reranking des meilleurs candidats.
    6. Retour des meilleurs passages.

    Cette classe n'effectue pas encore :

    - le Confidence Gate ;
    - la génération par LLM ;
    - la construction des citations finales.
    """

    def __init__(
        self,
        embedding_model_name: str = (
            DEFAULT_EMBEDDING_MODEL
        ),
        reranker_model_name: str = (
            DEFAULT_RERANKER_MODEL
        ),
        embedding_dimension: int = (
            DEFAULT_EMBEDDING_DIMENSION
        ),
        query_prefix: str = (
            DEFAULT_QUERY_PREFIX
        ),
        normalize_embeddings: bool = (
            DEFAULT_NORMALIZE_EMBEDDINGS
        ),
        device: str | None = None,
        embedding_device: str | None = None,
        reranker_device: str | None = None,
        candidate_k: int = (
            DEFAULT_CANDIDATE_K
        ),
        hybrid_top_k: int = (
            DEFAULT_HYBRID_TOP_K
        ),
        final_top_k: int = (
            DEFAULT_FINAL_TOP_K
        ),
        rrf_constant: int = (
            DEFAULT_RRF_CONSTANT
        ),
        embedding_batch_size: int = (
            DEFAULT_EMBEDDING_BATCH_SIZE
        ),
        reranker_batch_size: int = (
            DEFAULT_RERANKER_BATCH_SIZE
        ),
        max_length: int = (
            DEFAULT_MAX_LENGTH
        ),
        hybrid_retriever: (
            HybridRRFRetriever | None
        ) = None,
        reranker: (
            CrossEncoderReranker | None
        ) = None,
    ) -> None:
        """
        Initialise le pipeline.

        Les dépendances `hybrid_retriever` et `reranker`
        peuvent être injectées, ce qui facilite les tests et
        le remplacement futur d'un composant.
        """

        self.embedding_model_name = (
            self._validate_non_empty_text(
                embedding_model_name,
                "embedding_model_name",
            )
        )

        self.reranker_model_name = (
            self._validate_non_empty_text(
                reranker_model_name,
                "reranker_model_name",
            )
        )

        self.embedding_dimension = (
            self._validate_positive_integer(
                embedding_dimension,
                "embedding_dimension",
            )
        )

        self.query_prefix = str(
            query_prefix or ""
        )

        if not isinstance(
            normalize_embeddings,
            bool,
        ):
            raise TypeError(
                "normalize_embeddings doit être "
                "un booléen."
            )

        self.normalize_embeddings = (
            normalize_embeddings
        )

        self.candidate_k = (
            self._validate_positive_integer(
                candidate_k,
                "candidate_k",
            )
        )

        self.hybrid_top_k = (
            self._validate_positive_integer(
                hybrid_top_k,
                "hybrid_top_k",
            )
        )

        self.final_top_k = (
            self._validate_positive_integer(
                final_top_k,
                "final_top_k",
            )
        )

        self.embedding_batch_size = (
            self._validate_positive_integer(
                embedding_batch_size,
                "embedding_batch_size",
            )
        )

        self.reranker_batch_size = (
            self._validate_positive_integer(
                reranker_batch_size,
                "reranker_batch_size",
            )
        )

        self.max_length = (
            self._validate_positive_integer(
                max_length,
                "max_length",
            )
        )

        if self.hybrid_top_k > (
            self.candidate_k * 2
        ):
            raise ValueError(
                "hybrid_top_k est trop élevé par rapport "
                "au nombre total potentiel de candidats."
            )

        if self.final_top_k > (
            self.hybrid_top_k
        ):
            raise ValueError(
                "final_top_k ne peut pas être supérieur "
                "à hybrid_top_k."
            )

        self.rrf_constant = (
            self._validate_non_negative_integer(
                rrf_constant,
                "rrf_constant",
            )
        )

        # Compatibilité avec l'ancien paramètre unique `device`.
        fallback_device = (
            "auto"
            if device is None
            else str(device).strip().lower()
        )

        requested_embedding_device = (
            embedding_device
            if embedding_device is not None
            else fallback_device
        )

        requested_reranker_device = (
            reranker_device
            if reranker_device is not None
            else fallback_device
        )

        self.embedding_device = (
            self._resolve_device(
                requested_embedding_device
            )
        )

        self.reranker_device = (
            self._resolve_device(
                requested_reranker_device
            )
        )

        # Conservé pour compatibilité avec l'ancienne version.
        self.device = self.reranker_device

        self._embedding_model: (
            SentenceTransformer | None
        ) = None

        self.hybrid_retriever = (
            hybrid_retriever
            if hybrid_retriever is not None
            else HybridRRFRetriever(
                candidate_k=self.candidate_k,
                final_top_k=self.hybrid_top_k,
                rrf_constant=self.rrf_constant,
            )
        )

        self.reranker = (
            reranker
            if reranker is not None
            else CrossEncoderReranker(
                model_name=(
                    self.reranker_model_name
                ),
                device=self.reranker_device,
                batch_size=(
                    self.reranker_batch_size
                ),
                max_length=self.max_length,
            )
        )

    # ========================================================
    # Construction depuis la configuration YAML
    # ========================================================

    @classmethod
    def from_settings(
        cls,
        settings: (
            ApplicationSettings | None
        ) = None,
    ) -> "RetrievalPipeline":
        """
        Construit le pipeline à partir du fichier YAML.

        Si aucun objet `settings` n'est fourni, la fonction
        charge automatiquement `config/settings.yaml`.
        """

        application_settings = (
            settings
            if settings is not None
            else get_settings()
        )

        if not (
            application_settings
            .reranking
            .enabled
        ):
            raise ValueError(
                "Le pipeline actuel nécessite que "
                "reranking.enabled soit défini à true."
            )

        supported_vector_backends = {
            "pgvector",
        }

        supported_lexical_backends = {
            "sqlite_fts5",
        }

        vector_backend = (
            application_settings
            .retrieval
            .vector_backend
            .strip()
            .lower()
        )

        lexical_backend = (
            application_settings
            .retrieval
            .lexical_backend
            .strip()
            .lower()
        )

        if (
            vector_backend
            not in supported_vector_backends
        ):
            raise ValueError(
                "Backend vectoriel non pris en charge : "
                f"{vector_backend}. "
                "Backends disponibles : pgvector."
            )

        if (
            lexical_backend
            not in supported_lexical_backends
        ):
            raise ValueError(
                "Backend lexical non pris en charge : "
                f"{lexical_backend}. "
                "Backends disponibles : sqlite_fts5."
            )

        return cls(
            embedding_model_name=(
                application_settings
                .embedding
                .model_name
            ),
            reranker_model_name=(
                application_settings
                .reranking
                .model_name
            ),
            embedding_dimension=(
                application_settings
                .embedding
                .dimension
            ),
            query_prefix=(
                application_settings
                .embedding
                .query_prefix
            ),
            normalize_embeddings=(
                application_settings
                .embedding
                .normalize
            ),
            embedding_device=(
                application_settings
                .embedding
                .device
            ),
            reranker_device=(
                application_settings
                .reranking
                .device
            ),
            candidate_k=(
                application_settings
                .retrieval
                .candidate_k
            ),
            hybrid_top_k=(
                application_settings
                .retrieval
                .hybrid_top_k
            ),
            final_top_k=(
                application_settings
                .retrieval
                .final_top_k
            ),
            rrf_constant=(
                application_settings
                .retrieval
                .rrf_constant
            ),
            embedding_batch_size=(
                application_settings
                .embedding
                .batch_size
            ),
            reranker_batch_size=(
                application_settings
                .reranking
                .batch_size
            ),
            max_length=(
                application_settings
                .reranking
                .max_length
            ),
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_non_empty_text(
        value: str,
        field_name: str,
    ) -> str:
        normalized_value = str(
            value
        ).strip()

        if not normalized_value:
            raise ValueError(
                f"{field_name} ne peut pas être vide."
            )

        return normalized_value

    @staticmethod
    def _validate_positive_integer(
        value: int,
        field_name: str,
    ) -> int:
        try:
            normalized_value = int(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{field_name} doit être un entier."
            ) from error

        if normalized_value <= 0:
            raise ValueError(
                f"{field_name} doit être supérieur "
                "à zéro."
            )

        return normalized_value

    @staticmethod
    def _validate_non_negative_integer(
        value: int,
        field_name: str,
    ) -> int:
        try:
            normalized_value = int(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{field_name} doit être un entier."
            ) from error

        if normalized_value < 0:
            raise ValueError(
                f"{field_name} doit être positif "
                "ou nul."
            )

        return normalized_value

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        normalized_question = str(
            question
        ).strip()

        if not normalized_question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        return normalized_question

    @staticmethod
    def _validate_document_format(
        document_format: str | None,
    ) -> str | None:
        if document_format is None:
            return None

        normalized_format = (
            str(document_format)
            .strip()
            .lower()
        )

        allowed_formats = {
            "html",
            "markdown",
            "pdf",
        }

        if (
            normalized_format
            not in allowed_formats
        ):
            raise ValueError(
                "document_format doit être "
                "html, markdown ou pdf."
            )

        return normalized_format

    @staticmethod
    def _resolve_device(
        requested_device: str | None,
    ) -> str:
        """
        Convertit `auto` en périphérique réellement utilisable.
        """

        normalized_device = (
            "auto"
            if requested_device is None
            else str(
                requested_device
            ).strip().lower()
        )

        if normalized_device == "auto":
            return (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if normalized_device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "Le périphérique doit être "
                "auto, cpu ou cuda."
            )

        if (
            normalized_device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA est demandé mais aucun GPU "
                "compatible n'est disponible."
            )

        return normalized_device

    # ========================================================
    # État des modèles
    # ========================================================

    @property
    def embedding_model_loaded(
        self,
    ) -> bool:
        return (
            self._embedding_model
            is not None
        )

    @property
    def reranker_model_loaded(
        self,
    ) -> bool:
        return self.reranker.is_loaded

    # ========================================================
    # Gestion du modèle d'embedding
    # ========================================================

    def load_embedding_model(
        self,
    ) -> None:
        """
        Charge le modèle d'embedding une seule fois.
        """

        if (
            self._embedding_model
            is not None
        ):
            return

        self._embedding_model = (
            SentenceTransformer(
                self.embedding_model_name,
                device=self.embedding_device,
            )
        )

        # Compatibilité entre différentes versions
        # de sentence-transformers.
        if hasattr(
            self._embedding_model,
            "get_embedding_dimension",
        ):
            embedding_dimension = (
                self._embedding_model
                .get_embedding_dimension()
            )
        else:
            embedding_dimension = (
                self._embedding_model
                .get_sentence_embedding_dimension()
            )

        if (
            embedding_dimension
            != self.embedding_dimension
        ):
            self._embedding_model = None

            raise ValueError(
                "Le modèle d'embedding produit une "
                "dimension incompatible avec la base. "
                f"Dimension configurée : "
                f"{self.embedding_dimension}. "
                f"Dimension du modèle : "
                f"{embedding_dimension}."
            )

    def unload_models(
        self,
    ) -> None:
        """
        Libère les modèles neuronaux et la mémoire associée.
        """

        self._embedding_model = None
        self.reranker.unload_model()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Embedding de la question
    # ========================================================

    def embed_question(
        self,
        question: str,
    ) -> np.ndarray:
        """
        Encode une question avec le modèle configuré.
        """

        normalized_question = (
            self._validate_question(
                question
            )
        )

        self.load_embedding_model()

        if self._embedding_model is None:
            raise RuntimeError(
                "Le modèle d'embedding n'a pas "
                "été chargé."
            )

        embedding_input = (
            f"{self.query_prefix}"
            f"{normalized_question}"
        )

        embedding = (
            self._embedding_model.encode(
                [embedding_input],
                batch_size=(
                    self.embedding_batch_size
                ),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=(
                    self.normalize_embeddings
                ),
            )
        )

        embedding_array = np.asarray(
            embedding[0],
            dtype=np.float32,
        ).reshape(-1)

        expected_shape = (
            self.embedding_dimension,
        )

        if (
            embedding_array.shape
            != expected_shape
        ):
            raise RuntimeError(
                "L'embedding produit possède une "
                "dimension incorrecte. "
                f"Attendu : {expected_shape}. "
                f"Trouvé : "
                f"{embedding_array.shape}."
            )

        if not np.isfinite(
            embedding_array
        ).all():
            raise RuntimeError(
                "L'embedding contient des valeurs "
                "NaN ou infinies."
            )

        embedding_norm = float(
            np.linalg.norm(
                embedding_array
            )
        )

        if embedding_norm == 0.0:
            raise RuntimeError(
                "L'embedding possède une norme nulle."
            )

        return embedding_array

    # ========================================================
    # Exécution du pipeline
    # ========================================================

    def retrieve(
        self,
        question: str,
        document_format: str | None = None,
        final_top_k: int | None = None,
        lexical_question: str | None = None,
    ) -> RetrievalPipelineResponse:
        """
        Exécute le pipeline documentaire complet.

        Args:
            question:
                Question utilisateur.

            document_format:
                Filtre optionnel : html, markdown ou pdf.

            final_top_k:
                Nombre final de passages à retourner.
                Lorsque cette valeur est absente, la valeur
                configurée est utilisée.

        Returns:
            Une instance de RetrievalPipelineResponse.
        """

        normalized_question = (
            self._validate_question(
                question
            )
        )

        normalized_lexical_question = (
            normalized_question
            if lexical_question is None
            else self._validate_question(lexical_question)
        )

        normalized_format = (
            self._validate_document_format(
                document_format
            )
        )

        requested_final_top_k = (
            self.final_top_k
            if final_top_k is None
            else (
                self
                ._validate_positive_integer(
                    final_top_k,
                    "final_top_k",
                )
            )
        )

        if (
            requested_final_top_k
            > self.hybrid_top_k
        ):
            raise ValueError(
                "final_top_k ne peut pas dépasser "
                "hybrid_top_k."
            )

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # 1. Embedding
        # ----------------------------------------------------

        embedding_start = (
            time.perf_counter()
        )

        query_embedding = (
            self.embed_question(
                normalized_question
            )
        )

        embedding_time_ms = (
            time.perf_counter()
            - embedding_start
        ) * 1000

        # ----------------------------------------------------
        # 2. Hybrid Retrieval
        # ----------------------------------------------------

        hybrid_start = (
            time.perf_counter()
        )

        hybrid_results = (
            self.hybrid_retriever.search(
                question=(
                    normalized_lexical_question
                ),
                query_embedding=(
                    query_embedding
                ),
                candidate_k=(
                    self.candidate_k
                ),
                final_top_k=(
                    self.hybrid_top_k
                ),
                document_format=(
                    normalized_format
                ),
            )
        )

        hybrid_retrieval_time_ms = (
            time.perf_counter()
            - hybrid_start
        ) * 1000

        # Aucun candidat trouvé.
        if not hybrid_results:
            total_time_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            return self._build_response(
                question=(
                    normalized_question
                ),
                results=[],
                embedding_time_ms=(
                    embedding_time_ms
                ),
                hybrid_retrieval_time_ms=(
                    hybrid_retrieval_time_ms
                ),
                reranking_time_ms=0.0,
                total_time_ms=(
                    total_time_ms
                ),
                final_top_k=(
                    requested_final_top_k
                ),
                document_format=(
                    normalized_format
                ),
            )

        # ----------------------------------------------------
        # 3. Reranking
        # ----------------------------------------------------

        reranking_start = (
            time.perf_counter()
        )

        reranked_results = (
            self.reranker.rerank(
                question=(
                    normalized_question
                ),
                candidates=(
                    hybrid_results
                ),
                top_k=(
                    requested_final_top_k
                ),
            )
        )

        reranking_time_ms = (
            time.perf_counter()
            - reranking_start
        ) * 1000

        total_time_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        return self._build_response(
            question=(
                normalized_question
            ),
            results=(
                reranked_results
            ),
            embedding_time_ms=(
                embedding_time_ms
            ),
            hybrid_retrieval_time_ms=(
                hybrid_retrieval_time_ms
            ),
            reranking_time_ms=(
                reranking_time_ms
            ),
            total_time_ms=(
                total_time_ms
            ),
            final_top_k=(
                requested_final_top_k
            ),
            document_format=(
                normalized_format
            ),
        )

    # ========================================================
    # Construction de la réponse
    # ========================================================

    def _build_response(
        self,
        question: str,
        results: list[
            RerankedSearchResult
        ],
        embedding_time_ms: float,
        hybrid_retrieval_time_ms: float,
        reranking_time_ms: float,
        total_time_ms: float,
        final_top_k: int,
        document_format: str | None,
    ) -> RetrievalPipelineResponse:
        """
        Construit une réponse uniforme pour tous les cas.
        """

        timings = RetrievalTimings(
            embedding_time_ms=round(
                embedding_time_ms,
                4,
            ),
            hybrid_retrieval_time_ms=round(
                hybrid_retrieval_time_ms,
                4,
            ),
            reranking_time_ms=round(
                reranking_time_ms,
                4,
            ),
            total_time_ms=round(
                total_time_ms,
                4,
            ),
        )

        return RetrievalPipelineResponse(
            question=question,
            results=results,
            timings=timings,
            embedding_model=(
                self.embedding_model_name
            ),
            reranker_model=(
                self.reranker_model_name
            ),
            embedding_dimension=(
                self.embedding_dimension
            ),
            candidate_k=self.candidate_k,
            hybrid_top_k=(
                self.hybrid_top_k
            ),
            final_top_k=final_top_k,
            rrf_constant=(
                self.rrf_constant
            ),
            embedding_device=(
                self.embedding_device
            ),
            reranker_device=(
                self.reranker_device
            ),
            document_format_filter=(
                document_format
            ),
        )
