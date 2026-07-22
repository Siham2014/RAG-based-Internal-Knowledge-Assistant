from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.chunking.base import BaseChunker, ChunkingError
from src.chunking.block_builder import BlockBuilder
from src.chunking.models import (
    ChunkingConfig,
    ChunkingStrategy,
    DocumentChunk,
    TextBlock,
)
from src.chunking.tokenizer import (
    BaseTokenizer,
    HuggingFaceTokenizer,
    WhitespaceTokenizer,
)


class FixedSizeChunker(BaseChunker):
    """
    Chunker utilisant des fenêtres fixes de tokens.

    Exemple :
        chunk_size = 256
        chunk_overlap = 32
        step = 224

        chunk 0 : tokens 0 à 255
        chunk 1 : tokens 224 à 479
        chunk 2 : tokens 448 à 703

    Cette stratégie ne cherche pas à respecter les limites des phrases
    ou des paragraphes. Elle constitue la baseline du benchmark.
    """

    strategy_name = ChunkingStrategy.FIXED_SIZE.value

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        tokenizer: BaseTokenizer | None = None,
        block_builder: BlockBuilder | None = None,
    ) -> None:
        super().__init__(config)

        if config.strategy != ChunkingStrategy.FIXED_SIZE:
            raise ChunkingError(
                "FixedSizeChunker nécessite une configuration "
                "avec strategy='fixed_size'."
            )

        self.tokenizer = tokenizer or self._create_default_tokenizer()
        self.block_builder = block_builder or BlockBuilder()

    def chunk(self, document: Any) -> list[DocumentChunk]:
        """
        Transforme un document ou une liste de TextBlock en chunks fixes.

        Entrées acceptées :
        - un document compatible avec BlockBuilder ;
        - une liste de TextBlock.
        """

        self.validate_document(document)

        blocks = self._resolve_blocks(document)

        return self.chunk_blocks(blocks)

    def chunk_blocks(
        self,
        blocks: Sequence[TextBlock],
    ) -> list[DocumentChunk]:
        """
        Découpe directement une séquence de TextBlock.
        """

        validated_blocks = self._validate_blocks(blocks)

        token_ids, token_block_indexes = self._build_token_stream(
            validated_blocks
        )

        if not token_ids:
            raise ChunkingError(
                "Les blocs ne contiennent aucun token exploitable."
            )

        chunks: list[DocumentChunk] = []

        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        step = chunk_size - chunk_overlap

        start_token = 0
        chunk_index = 0

        while start_token < len(token_ids):
            end_token = min(
                start_token + chunk_size,
                len(token_ids),
            )

            current_token_ids = token_ids[
                start_token:end_token
            ]

            current_block_indexes = token_block_indexes[
                start_token:end_token
            ]

            chunk_text = self.tokenizer.decode(
                current_token_ids
            ).strip()

            if chunk_text:
                covered_blocks = self._get_covered_blocks(
                    blocks=validated_blocks,
                    block_indexes=current_block_indexes,
                )

                document_chunk = self._build_document_chunk(
                    chunk_text=chunk_text,
                    token_count=len(current_token_ids),
                    chunk_index=chunk_index,
                    start_token=start_token,
                    end_token=end_token,
                    covered_blocks=covered_blocks,
                )

                chunks.append(document_chunk)
                chunk_index += 1

            if end_token >= len(token_ids):
                break

            start_token += step

        if not chunks:
            raise ChunkingError(
                "Le FixedSizeChunker n'a produit aucun chunk."
            )

        return chunks

    def _resolve_blocks(
        self,
        document: Any,
    ) -> list[TextBlock]:
        """
        Détermine si l'entrée contient déjà des TextBlock.
        """

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

    def _validate_blocks(
        self,
        blocks: Sequence[TextBlock],
    ) -> list[TextBlock]:
        if not isinstance(blocks, Sequence):
            raise ChunkingError(
                "blocks doit être une séquence de TextBlock."
            )

        validated_blocks = list(blocks)

        if not validated_blocks:
            raise ChunkingError(
                "La liste des blocs ne peut pas être vide."
            )

        if not all(
            isinstance(block, TextBlock)
            for block in validated_blocks
        ):
            raise ChunkingError(
                "Tous les éléments doivent être des TextBlock."
            )

        document_ids = {
            block.document_id
            for block in validated_blocks
        }

        if len(document_ids) != 1:
            raise ChunkingError(
                "Tous les blocs doivent appartenir au même document."
            )

        return validated_blocks

    def _build_token_stream(
        self,
        blocks: Sequence[TextBlock],
    ) -> tuple[list[int], list[int]]:
        """
        Construit un seul flux de tokens pour le document.

        token_block_indexes permet de retrouver les blocs d'origine
        couverts par chaque fenêtre.
        """

        token_ids: list[int] = []
        token_block_indexes: list[int] = []

        for block_position, block in enumerate(blocks):
            current_ids = self.tokenizer.encode(
                block.text
            )

            if not current_ids:
                continue

            token_ids.extend(current_ids)

            token_block_indexes.extend(
                [block_position] * len(current_ids)
            )

        return token_ids, token_block_indexes

    @staticmethod
    def _get_covered_blocks(
        *,
        blocks: Sequence[TextBlock],
        block_indexes: Sequence[int],
    ) -> list[TextBlock]:
        """
        Retourne les blocs traversés par une fenêtre sans doublon.
        """

        unique_indexes: list[int] = []

        for block_index in block_indexes:
            if block_index not in unique_indexes:
                unique_indexes.append(block_index)

        return [
            blocks[block_index]
            for block_index in unique_indexes
        ]

    def _build_document_chunk(
        self,
        *,
        chunk_text: str,
        token_count: int,
        chunk_index: int,
        start_token: int,
        end_token: int,
        covered_blocks: Sequence[TextBlock],
    ) -> DocumentChunk:
        first_block = covered_blocks[0]

        section_ids = self._unique_non_empty(
            block.section_id
            for block in covered_blocks
        )

        section_headings = self._unique_non_empty(
            block.section_heading
            for block in covered_blocks
        )

        content_types = self._unique_non_empty(
            block.block_type
            for block in covered_blocks
        )

        heading_path = self._common_heading_path(
            [
                block.heading_path
                for block in covered_blocks
            ]
        )

        section_id = (
            section_ids[0]
            if len(section_ids) == 1
            else None
        )

        section_heading = (
            section_headings[0]
            if len(section_headings) == 1
            else None
        )

        metadata = {
            "tokenizer_name": self.tokenizer.name,
            "start_token": start_token,
            "end_token": end_token,
            "covered_block_ids": [
                block.block_id
                for block in covered_blocks
            ],
            "covered_block_indexes": [
                block.block_index
                for block in covered_blocks
            ],
            "section_ids": section_ids,
            "section_headings": section_headings,
            "fixed_size_step": (
                self.config.chunk_size
                - self.config.chunk_overlap
            ),
        }

        return DocumentChunk.create(
            document_id=first_block.document_id,
            text=chunk_text,
            strategy=self.config.strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunk_index=chunk_index,
            token_count=token_count,
            source_url=first_block.source_url,
            document_title=first_block.document_title,
            section_id=section_id,
            section_heading=section_heading,
            heading_path=heading_path,
            content_types=content_types,
            metadata=metadata,
        )

    def _create_default_tokenizer(
        self,
    ) -> BaseTokenizer:
        """
        Crée le tokenizer défini dans ChunkingConfig.

        'default' ou 'whitespace' est réservé aux tests.
        Pour le benchmark réel, tokenizer_name contiendra par exemple :
        BAAI/bge-small-en-v1.5
        """

        tokenizer_name = self.config.tokenizer_name.strip()

        if tokenizer_name in {
            "",
            "default",
            "whitespace",
        }:
            return WhitespaceTokenizer()

        return HuggingFaceTokenizer(
            tokenizer_name
        )

    @staticmethod
    def _unique_non_empty(
        values: Any,
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
        """
        Retourne le préfixe commun des chemins de titres.

        Exemple :
            ["Azure", "Reliability", "Zones"]
            ["Azure", "Reliability", "Regions"]

        Résultat :
            ["Azure", "Reliability"]
        """

        if not paths:
            return []

        common_path = list(paths[0])

        for path in paths[1:]:
            maximum_length = min(
                len(common_path),
                len(path),
            )

            common_length = 0

            while (
                common_length < maximum_length
                and common_path[common_length]
                == path[common_length]
            ):
                common_length += 1

            common_path = common_path[:common_length]

            if not common_path:
                break

        return common_path