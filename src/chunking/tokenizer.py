from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class TokenizerError(RuntimeError):
    """
    Erreur déclenchée pendant une opération de tokenisation.
    """


@dataclass(slots=True, frozen=True)
class TokenWindow:
    """
    Représente une fenêtre de tokens.

    start_token est inclus.
    end_token est exclu.
    """

    token_ids: list[int]
    start_token: int
    end_token: int

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class BaseTokenizer(ABC):
    """
    Contrat commun aux tokenizers du module de chunking.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Transforme un texte en liste d'identifiants de tokens.
        """

        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Reconstruit un texte à partir d'identifiants de tokens.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nom du tokenizer.
        """

        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """
        Compte le nombre de tokens d'un texte.
        """

        if not isinstance(text, str):
            raise TypeError("text doit être une chaîne de caractères.")

        if not text:
            return 0

        return len(self.encode(text))

    def split_token_windows(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int = 0,
    ) -> list[TokenWindow]:
        """
        Découpe un texte en fenêtres de tokens avec overlap.

        Exemple :
        chunk_size = 256
        chunk_overlap = 32
        step = 224
        """

        self._validate_window_parameters(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        token_ids = self.encode(text)

        if not token_ids:
            return []

        step = chunk_size - chunk_overlap
        windows: list[TokenWindow] = []

        start = 0

        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            current_ids = token_ids[start:end]

            windows.append(
                TokenWindow(
                    token_ids=list(current_ids),
                    start_token=start,
                    end_token=end,
                )
            )

            if end >= len(token_ids):
                break

            start += step

        return windows

    def split_text_by_tokens(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int = 0,
    ) -> list[str]:
        """
        Découpe un texte et retourne directement les textes reconstruits.
        """

        windows = self.split_token_windows(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks: list[str] = []

        for window in windows:
            decoded = self.decode(window.token_ids).strip()

            if decoded:
                chunks.append(decoded)

        return chunks

    def take_first_tokens(
        self,
        text: str,
        token_count: int,
    ) -> str:
        """
        Retourne les premiers tokens d'un texte.
        """

        if token_count < 0:
            raise TokenizerError(
                "token_count ne peut pas être négatif."
            )

        if token_count == 0:
            return ""

        token_ids = self.encode(text)

        return self.decode(token_ids[:token_count]).strip()

    def take_last_tokens(
        self,
        text: str,
        token_count: int,
    ) -> str:
        """
        Retourne les derniers tokens d'un texte.
        """

        if token_count < 0:
            raise TokenizerError(
                "token_count ne peut pas être négatif."
            )

        if token_count == 0:
            return ""

        token_ids = self.encode(text)

        return self.decode(token_ids[-token_count:]).strip()

    def truncate(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """
        Tronque un texte à une limite maximale de tokens.
        """

        if max_tokens <= 0:
            raise TokenizerError(
                "max_tokens doit être strictement supérieur à zéro."
            )

        token_ids = self.encode(text)

        if len(token_ids) <= max_tokens:
            return text.strip()

        return self.decode(token_ids[:max_tokens]).strip()

    @staticmethod
    def _validate_window_parameters(
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if chunk_size <= 0:
            raise TokenizerError(
                "chunk_size doit être strictement supérieur à zéro."
            )

        if chunk_overlap < 0:
            raise TokenizerError(
                "chunk_overlap ne peut pas être négatif."
            )

        if chunk_overlap >= chunk_size:
            raise TokenizerError(
                "chunk_overlap doit être strictement inférieur "
                "à chunk_size."
            )


class WhitespaceTokenizer(BaseTokenizer):
    """
    Tokenizer simple et déterministe utilisé pour les tests unitaires.

    Chaque mot séparé par un espace représente un token.

    Il ne sera pas utilisé pour le benchmark final.
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "whitespace"

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text doit être une chaîne de caractères.")

        words = text.split()
        token_ids: list[int] = []

        for word in words:
            if word not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[word] = token_id
                self._id_to_token[token_id] = word

            token_ids.append(self._token_to_id[word])

        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        words: list[str] = []

        for token_id in token_ids:
            try:
                words.append(self._id_to_token[int(token_id)])
            except KeyError as exc:
                raise TokenizerError(
                    f"Token inconnu : {token_id}."
                ) from exc

        return " ".join(words)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Tokenizer basé sur Hugging Face Transformers.

    Il sera utilisé pour le benchmark réel avec le tokenizer du modèle
    d'embedding, par exemple BAAI/bge-small-en-v1.5.
    """

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = False,
        use_fast: bool = True,
    ) -> None:
        if not model_name or not model_name.strip():
            raise TokenizerError(
                "model_name ne peut pas être vide."
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TokenizerError(
                "Le package transformers n'est pas installé. "
                "Exécutez : python -m pip install transformers tokenizers"
            ) from exc

        self._model_name = model_name.strip()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=local_files_only,
                use_fast=use_fast,
            )
        except Exception as exc:
            raise TokenizerError(
                f"Impossible de charger le tokenizer "
                f"{self._model_name!r} : {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text doit être une chaîne de caractères.")

        if not text:
            return []

        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""

        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )