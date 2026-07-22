from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from src.chunking.base import BaseChunker, ChunkingError
from src.chunking.block_builder import BlockBuilder
from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentChunk,
    TextBlock,
)
from src.chunking.semantic_encoder import (
    BaseSemanticEncoder,
    SentenceTransformerEncoder,
    cosine_similarity,
)
from src.chunking.tokenizer import (
    BaseTokenizer,
    HuggingFaceTokenizer,
    WhitespaceTokenizer,
)


@dataclass(slots=True)
class SemanticUnit:
    """Phrase ou segment atomique associé à son bloc d'origine."""

    text: str
    block: TextBlock
    token_count: int
    embedding_index: int | None = None


@dataclass(slots=True)
class SemanticAssembly:
    """Chunk sémantique interne avant création du DocumentChunk."""

    text: str
    units: list[SemanticUnit]
    overlap_token_count: int
    boundary_reason: str
    minimum_similarity: float | None


class SemanticChunker(BaseChunker):
    """
    Découpe les documents selon les changements de sujet.

    Une frontière est créée lorsque la similarité entre deux unités voisines
    devient inférieure à semantic_similarity_threshold. La taille maximale
    chunk_size reste toujours prioritaire.
    """

    strategy_name = ChunkingStrategy.SEMANTIC.value

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        encoder: BaseSemanticEncoder | None = None,
        tokenizer: BaseTokenizer | None = None,
        block_builder: BlockBuilder | None = None,
    ) -> None:
        super().__init__(config)

        if config.strategy != ChunkingStrategy.SEMANTIC:
            raise ChunkingError(
                "SemanticChunker nécessite une configuration "
                "avec strategy='semantic'."
            )

        threshold = config.semantic_similarity_threshold

        if threshold is None:
            threshold = 0.65

        if not -1.0 <= threshold <= 1.0:
            raise ChunkingError(
                "semantic_similarity_threshold doit être compris "
                "entre -1 et 1."
            )

        self.similarity_threshold = float(threshold)
        self.encoder = encoder or SentenceTransformerEncoder()
        self.tokenizer = tokenizer or self._create_default_tokenizer()
        self.block_builder = block_builder or BlockBuilder()

    def chunk(self, document: Any) -> list[DocumentChunk]:
        self.validate_document(document)
        return self.chunk_blocks(self._resolve_blocks(document))

    def chunk_blocks(
        self,
        blocks: Sequence[TextBlock],
    ) -> list[DocumentChunk]:
        validated_blocks = self._validate_blocks(blocks)
        units = self._create_units(validated_blocks)

        if not units:
            raise ChunkingError(
                "Le SemanticChunker n'a produit aucune unité."
            )

        embeddings = self.encoder.encode(
            [unit.text for unit in units]
        )

        if embeddings.shape[0] != len(units):
            raise ChunkingError(
                "Le nombre d'embeddings ne correspond pas "
                "au nombre d'unités."
            )

        for index, unit in enumerate(units):
            unit.embedding_index = index

        similarities = [
            cosine_similarity(
                embeddings[index],
                embeddings[index + 1],
            )
            for index in range(len(units) - 1)
        ]

        assemblies = self._assemble_chunks(
            units=units,
            similarities=similarities,
        )

        chunks = [
            self._build_document_chunk(
                assembly=assembly,
                chunk_index=index,
            )
            for index, assembly in enumerate(assemblies)
        ]

        if not chunks:
            raise ChunkingError(
                "Le SemanticChunker n'a produit aucun chunk."
            )

        return chunks

    def _create_units(
        self,
        blocks: Sequence[TextBlock],
    ) -> list[SemanticUnit]:
        units: list[SemanticUnit] = []

        for block in blocks:
            units.extend(self._split_block(block))

        return units

    def _split_block(
        self,
        block: TextBlock,
    ) -> list[SemanticUnit]:
        text = block.text.strip()

        if not text:
            return []

        protected = (
            block.block_type == "code"
            and self.config.preserve_code_blocks
        ) or (
            block.block_type == "table"
            and self.config.preserve_tables
        )

        if protected:
            return self._split_text_by_tokens(text, block)

        sentences = self._split_sentences(text)
        units: list[SemanticUnit] = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if (
                self.tokenizer.count_tokens(sentence)
                > self.config.chunk_size
            ):
                units.extend(
                    self._split_text_by_tokens(sentence, block)
                )
            else:
                units.append(
                    SemanticUnit(
                        text=sentence,
                        block=block,
                        token_count=self.tokenizer.count_tokens(
                            sentence
                        ),
                    )
                )

        return units

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Sépare les phrases tout en conservant leur ponctuation finale.

        Les titres sans ponctuation restent des unités indépendantes,
        car chaque TextBlock est traité séparément.
        """

        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _split_text_by_tokens(
        self,
        text: str,
        block: TextBlock,
    ) -> list[SemanticUnit]:
        windows = self.tokenizer.split_token_windows(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=0,
        )

        result: list[SemanticUnit] = []

        for window in windows:
            decoded = self.tokenizer.decode(window.token_ids).strip()

            if not decoded:
                continue

            result.append(
                SemanticUnit(
                    text=decoded,
                    block=block,
                    token_count=len(window.token_ids),
                )
            )

        return result

    def _assemble_chunks(
        self,
        *,
        units: Sequence[SemanticUnit],
        similarities: Sequence[float],
    ) -> list[SemanticAssembly]:
        assemblies: list[SemanticAssembly] = []

        current_units: list[SemanticUnit] = []
        current_parts: list[str] = []
        current_overlap = 0
        current_similarities: list[float] = []

        for index, unit in enumerate(units):
            candidate_text = self._combine_parts(
                [*current_parts, unit.text]
            )
            candidate_count = self.tokenizer.count_tokens(
                candidate_text
            )

            similarity_before = (
                similarities[index - 1]
                if index > 0
                else None
            )

            semantic_break = (
                bool(current_parts)
                and similarity_before is not None
                and similarity_before < self.similarity_threshold
                and self._current_content_token_count(current_parts)
                >= self.config.min_chunk_size
            )

            size_break = (
                bool(current_parts)
                and candidate_count > self.config.chunk_size
            )

            if semantic_break or size_break:
                reason = (
                    "semantic_threshold"
                    if semantic_break
                    else "chunk_size"
                )

                assemblies.append(
                    self._create_assembly(
                        text=self._combine_parts(current_parts),
                        units=current_units,
                        overlap_token_count=current_overlap,
                        boundary_reason=reason,
                        similarities=current_similarities,
                    )
                )

                previous_text = self._combine_parts(current_parts)
                overlap_text = self._get_overlap_text(previous_text)

                current_units = []
                current_parts = []
                current_overlap = 0
                current_similarities = []

                if overlap_text:
                    with_overlap = self._combine_parts(
                        [overlap_text, unit.text]
                    )

                    if (
                        self.tokenizer.count_tokens(with_overlap)
                        <= self.config.chunk_size
                    ):
                        current_parts.append(overlap_text)
                        current_overlap = (
                            self.tokenizer.count_tokens(overlap_text)
                        )

                current_units.append(unit)
                current_parts.append(unit.text)

                if (
                    self.tokenizer.count_tokens(
                        self._combine_parts(current_parts)
                    )
                    > self.config.chunk_size
                ):
                    raise ChunkingError(
                        "Erreur interne : un chunk sémantique "
                        "dépasse chunk_size."
                    )

                continue

            if current_units and similarity_before is not None:
                current_similarities.append(similarity_before)

            current_units.append(unit)
            current_parts.append(unit.text)

        if current_parts:
            assemblies.append(
                self._create_assembly(
                    text=self._combine_parts(current_parts),
                    units=current_units,
                    overlap_token_count=current_overlap,
                    boundary_reason="document_end",
                    similarities=current_similarities,
                )
            )

        return self._remove_duplicates(assemblies)

    def _current_content_token_count(
        self,
        parts: Sequence[str],
    ) -> int:
        return self.tokenizer.count_tokens(
            self._combine_parts(parts)
        )

    def _create_assembly(
        self,
        *,
        text: str,
        units: Sequence[SemanticUnit],
        overlap_token_count: int,
        boundary_reason: str,
        similarities: Sequence[float],
    ) -> SemanticAssembly:
        normalized = text.strip()
        token_count = self.tokenizer.count_tokens(normalized)

        if not normalized:
            raise ChunkingError(
                "Un chunk sémantique ne peut pas être vide."
            )

        if token_count > self.config.chunk_size:
            raise ChunkingError(
                "Un chunk sémantique dépasse chunk_size "
                f"({token_count} > {self.config.chunk_size})."
            )

        minimum_similarity = (
            min(similarities)
            if similarities
            else None
        )

        return SemanticAssembly(
            text=normalized,
            units=list(units),
            overlap_token_count=overlap_token_count,
            boundary_reason=boundary_reason,
            minimum_similarity=minimum_similarity,
        )

    def _get_overlap_text(self, text: str) -> str:
        if self.config.chunk_overlap == 0:
            return ""

        return self.tokenizer.take_last_tokens(
            text,
            self.config.chunk_overlap,
        )

    @staticmethod
    def _combine_parts(parts: Sequence[str]) -> str:
        return " ".join(
            part.strip()
            for part in parts
            if part.strip()
        ).strip()

    @staticmethod
    def _remove_duplicates(
        assemblies: Sequence[SemanticAssembly],
    ) -> list[SemanticAssembly]:
        result: list[SemanticAssembly] = []
        previous_text: str | None = None

        for assembly in assemblies:
            if assembly.text == previous_text:
                continue

            result.append(assembly)
            previous_text = assembly.text

        return result

    def _build_document_chunk(
        self,
        *,
        assembly: SemanticAssembly,
        chunk_index: int,
    ) -> DocumentChunk:
        blocks = self._unique_blocks(
            unit.block for unit in assembly.units
        )

        if not blocks:
            raise ChunkingError(
                "Impossible de déterminer les blocs du chunk."
            )

        first_block = blocks[0]

        section_ids = self._unique_non_empty(
            block.section_id for block in blocks
        )
        section_headings = self._unique_non_empty(
            block.section_heading for block in blocks
        )
        content_types = self._unique_non_empty(
            block.block_type for block in blocks
        )

        metadata = {
            "tokenizer_name": self.tokenizer.name,
            "semantic_encoder": self.encoder.model_name,
            "semantic_similarity_threshold": (
                self.similarity_threshold
            ),
            "minimum_internal_similarity": (
                assembly.minimum_similarity
            ),
            "boundary_reason": assembly.boundary_reason,
            "overlap_token_count": (
                assembly.overlap_token_count
            ),
            "covered_block_ids": [
                block.block_id for block in blocks
            ],
            "covered_block_indexes": [
                block.block_index for block in blocks
            ],
            "section_ids": section_ids,
            "section_headings": section_headings,
        }

        return DocumentChunk.create(
            document_id=first_block.document_id,
            text=assembly.text,
            strategy=self.config.strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunk_index=chunk_index,
            token_count=self.tokenizer.count_tokens(
                assembly.text
            ),
            source_url=first_block.source_url,
            document_title=first_block.document_title,
            section_id=(
                section_ids[0]
                if len(section_ids) == 1
                else None
            ),
            section_heading=(
                section_headings[0]
                if len(section_headings) == 1
                else None
            ),
            heading_path=self._common_heading_path(
                [block.heading_path for block in blocks]
            ),
            content_types=content_types,
            metadata=metadata,
        )

    def _resolve_blocks(self, document: Any) -> list[TextBlock]:
        if isinstance(document, Sequence) and not isinstance(
            document,
            (str, bytes),
        ):
            items = list(document)

            if items and all(
                isinstance(item, TextBlock)
                for item in items
            ):
                return items

        return self.block_builder.build(document)

    @staticmethod
    def _validate_blocks(
        blocks: Sequence[TextBlock],
    ) -> list[TextBlock]:
        validated = list(blocks)

        if not validated:
            raise ChunkingError(
                "La liste des blocs ne peut pas être vide."
            )

        if not all(
            isinstance(block, TextBlock)
            for block in validated
        ):
            raise ChunkingError(
                "Tous les éléments doivent être des TextBlock."
            )

        if len(
            {block.document_id for block in validated}
        ) != 1:
            raise ChunkingError(
                "Tous les blocs doivent appartenir au même document."
            )

        return validated

    @staticmethod
    def _unique_blocks(
        blocks: Iterable[TextBlock],
    ) -> list[TextBlock]:
        result: list[TextBlock] = []
        seen: set[str] = set()

        for block in blocks:
            if block.block_id in seen:
                continue

            seen.add(block.block_id)
            result.append(block)

        return result

    @staticmethod
    def _unique_non_empty(
        values: Iterable[Any],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            if value is None:
                continue

            normalized = str(value).strip()

            if normalized and normalized not in result:
                result.append(normalized)

        return result

    @staticmethod
    def _common_heading_path(
        paths: Sequence[Sequence[str]],
    ) -> list[str]:
        if not paths:
            return []

        common = list(paths[0])

        for path in paths[1:]:
            common_length = 0

            for left, right in zip(common, path):
                if left != right:
                    break
                common_length += 1

            common = common[:common_length]

            if not common:
                break

        return common

    def _create_default_tokenizer(self) -> BaseTokenizer:
        name = self.config.tokenizer_name.strip()

        if name in {"", "default", "whitespace"}:
            return WhitespaceTokenizer()

        return HuggingFaceTokenizer(name)