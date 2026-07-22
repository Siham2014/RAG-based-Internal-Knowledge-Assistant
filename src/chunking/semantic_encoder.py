from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class SemanticEncoderError(RuntimeError):
    """Erreur produite pendant la génération des embeddings sémantiques."""


class BaseSemanticEncoder(ABC):
    """Contrat commun des encodeurs utilisés par SemanticChunker."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nom public du modèle d'embedding."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Retourne une matrice de forme (nombre_textes, dimension).

        Les vecteurs doivent être normalisés afin que leur produit scalaire
        corresponde à la similarité cosinus.
        """


class SentenceTransformerEncoder(BaseSemanticEncoder):
    """
    Encodeur basé sur sentence-transformers.

    Le modèle est chargé de manière paresseuse au premier appel à encode().
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name ne peut pas être vide.")

        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif.")

        self._model_name = model_name.strip()
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticEncoderError(
                "La dépendance 'sentence-transformers' est absente. "
                "Installe-la avec : pip install sentence-transformers numpy"
            ) from exc

        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=self.device,
            )
        except Exception as exc:
            raise SemanticEncoderError(
                f"Impossible de charger le modèle '{self._model_name}'."
            ) from exc

        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = [str(text).strip() for text in texts]

        if not cleaned:
            return np.empty((0, 0), dtype=np.float32)

        if any(not text for text in cleaned):
            raise SemanticEncoderError(
                "Tous les textes à encoder doivent être non vides."
            )

        model = self._load_model()

        try:
            embeddings = model.encode(
                cleaned,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise SemanticEncoderError(
                "La génération des embeddings a échoué."
            ) from exc

        matrix = np.asarray(embeddings, dtype=np.float32)

        if matrix.ndim != 2 or matrix.shape[0] != len(cleaned):
            raise SemanticEncoderError(
                "La forme des embeddings retournés est invalide."
            )

        if not self.normalize_embeddings:
            matrix = normalize_rows(matrix)

        return matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque vecteur selon sa norme L2."""

    array = np.asarray(matrix, dtype=np.float32)

    if array.ndim != 2:
        raise SemanticEncoderError(
            "La matrice d'embeddings doit avoir deux dimensions."
        )

    norms = np.linalg.norm(array, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise SemanticEncoderError(
            "Un embedding nul ne peut pas être normalisé."
        )

    return array / norms


def cosine_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """
    Similarité cosinus entre deux vecteurs.

    Les vecteurs sont normalisés ici pour accepter aussi les encodeurs de test.
    """

    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)

    if left_vector.shape != right_vector.shape:
        raise SemanticEncoderError(
            "Les deux embeddings doivent avoir la même dimension."
        )

    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))

    if left_norm == 0.0 or right_norm == 0.0:
        raise SemanticEncoderError(
            "La similarité cosinus est impossible avec un vecteur nul."
        )

    similarity = float(
        np.dot(left_vector, right_vector)
        / (left_norm * right_norm)
    )

    return max(-1.0, min(1.0, similarity))