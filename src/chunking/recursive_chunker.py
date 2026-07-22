from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable

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


@dataclass(slots=True)
class RecursiveUnit:
    """
    Petite unité textuelle produite par le découpage récursif.

    Chaque unité conserve le bloc documentaire d'origine afin de
    préserver les métadonnées lors de la création des chunks finaux.
    """

    text: str
    block: TextBlock
    token_count: int


@dataclass(slots=True)
class ChunkAssembly:
    """
    Représentation interne d'un chunk avant création du DocumentChunk.
    """

    text: str
    units: list[RecursiveUnit]
    overlap_token_count: int = 0


class RecursiveChunker(BaseChunker):
    """
    Chunker récursif respectant autant que possible les limites naturelles.

    Ordre de découpage :
    1. doubles sauts de ligne ;
    2. sauts de ligne ;
    3. fins de phrases ;
    4. point-virgule ;
    5. virgule ;
    6. espaces ;
    7. découpage strict en tokens.

    L'overlap est ajouté entre deux chunks lorsque cela ne fait pas
    dépasser la taille maximale définie dans la configuration.
    """

    strategy_name = ChunkingStrategy.RECURSIVE.value

    DEFAULT_SEPARATORS = (
        "\n\n",
        "\n",
        "sentence",
        "; ",
        ", ",
        " ",
    )

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        tokenizer: BaseTokenizer | None = None,
        block_builder: BlockBuilder | None = None,
        separators: Sequence[str] | None = None,
    ) -> None:
        super().__init__(config)

        if config.strategy != ChunkingStrategy.RECURSIVE:
            raise ChunkingError(
                "RecursiveChunker nécessite une configuration "
                "avec strategy='recursive'."
            )

        self.tokenizer = tokenizer or self._create_default_tokenizer()
        self.block_builder = block_builder or BlockBuilder()
        self.separators = tuple(separators or self.DEFAULT_SEPARATORS)

        if not self.separators:
            raise ChunkingError(
                "La liste des séparateurs ne peut pas être vide."
            )

    def chunk(self, document: Any) -> list[DocumentChunk]:
        """
        Transforme un document complet ou une séquence de TextBlock en chunks.
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

        units: list[RecursiveUnit] = []

        for block in validated_blocks:
            units.extend(self._split_block(block))

        if not units:
            raise ChunkingError(
                "Le RecursiveChunker n'a produit aucune unité."
            )

        assemblies = self._assemble_chunks(units)

        chunks = [
            self._build_document_chunk(
                assembly=assembly,
                chunk_index=index,
            )
            for index, assembly in enumerate(assemblies)
        ]

        if not chunks:
            raise ChunkingError(
                "Le RecursiveChunker n'a produit aucun chunk."
            )

        return chunks

    def _split_block(
        self,
        block: TextBlock,
    ) -> list[RecursiveUnit]:
        """
        Découpe un bloc seulement lorsqu'il dépasse chunk_size.
        """

        token_count = self.tokenizer.count_tokens(block.text)

        if token_count == 0:
            return []

        if token_count <= self.config.chunk_size:
            return [
                RecursiveUnit(
                    text=block.text.strip(),
                    block=block,
                    token_count=token_count,
                )
            ]

        preserve_block = (
            block.block_type == "code"
            and self.config.preserve_code_blocks
        ) or (
            block.block_type == "table"
            and self.config.preserve_tables
        )

        if preserve_block:
            return self._split_by_tokens(
                text=block.text,
                block=block,
            )

        parts = self._recursive_split(
            text=block.text,
            separator_index=0,
        )

        units: list[RecursiveUnit] = []

        for part in parts:
            normalized = part.strip()

            if not normalized:
                continue

            count = self.tokenizer.count_tokens(normalized)

            if count > self.config.chunk_size:
                units.extend(
                    self._split_by_tokens(
                        text=normalized,
                        block=block,
                    )
                )
            else:
                units.append(
                    RecursiveUnit(
                        text=normalized,
                        block=block,
                        token_count=count,
                    )
                )

        return units

    def _recursive_split(
        self,
        *,
        text: str,
        separator_index: int,
    ) -> list[str]:
        """
        Essaie successivement les séparateurs hiérarchiques.

        Si un segment reste trop grand, le séparateur suivant est utilisé.
        En dernier recours, le texte est découpé strictement par tokens.
        """

        normalized_text = text.strip()

        if not normalized_text:
            return []

        token_count = self.tokenizer.count_tokens(normalized_text)

        if token_count <= self.config.chunk_size:
            return [normalized_text]

        if separator_index >= len(self.separators):
            return self._token_text_parts(normalized_text)

        separator = self.separators[separator_index]

        raw_parts = self._split_with_separator(
            normalized_text,
            separator,
        )

        if len(raw_parts) <= 1:
            return self._recursive_split(
                text=normalized_text,
                separator_index=separator_index + 1,
            )

        results: list[str] = []

        for raw_part in raw_parts:
            part = raw_part.strip()

            if not part:
                continue

            if self.tokenizer.count_tokens(part) <= self.config.chunk_size:
                results.append(part)
            else:
                results.extend(
                    self._recursive_split(
                        text=part,
                        separator_index=separator_index + 1,
                    )
                )

        return results

    def _split_with_separator(
        self,
        text: str,
        separator: str,
    ) -> list[str]:
        """
        Découpe le texte en conservant autant que possible la ponctuation.
        """

        if separator == "sentence":
            if not self.config.preserve_sentences:
                return [text]

            return self._split_sentences(text)

        if separator == " ":
            return text.split()

        if separator in {"; ", ", "}:
            punctuation = separator.strip()
            pattern = rf"(?<={re.escape(punctuation)})\s+"
            return re.split(pattern, text)

        return text.split(separator)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Découpe après les ponctuations de fin de phrase.
        """

        return re.split(r"(?<=[.!?])\s+", text)

    def _token_text_parts(self, text: str) -> list[str]:
        """
        Dernier recours : découpage strict selon chunk_size.
        """

        windows = self.tokenizer.split_token_windows(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=0,
        )

        return [
            self.tokenizer.decode(window.token_ids).strip()
            for window in windows
            if window.token_ids
        ]

    def _split_by_tokens(
        self,
        *,
        text: str,
        block: TextBlock,
    ) -> list[RecursiveUnit]:
        """
        Découpe un bloc trop grand en fenêtres sans overlap.

        L'overlap est ajouté pendant l'assemblage final.
        """

        windows = self.tokenizer.split_token_windows(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=0,
        )

        units: list[RecursiveUnit] = []

        for window in windows:
            decoded = self.tokenizer.decode(window.token_ids).strip()

            if not decoded:
                continue

            units.append(
                RecursiveUnit(
                    text=decoded,
                    block=block,
                    token_count=len(window.token_ids),
                )
            )

        return units

    def _assemble_chunks(
        self,
        units: Sequence[RecursiveUnit],
    ) -> list[ChunkAssembly]:
        """
        Fusionne les unités sans dépasser chunk_size.

        Quand le chunk courant est terminé, les derniers tokens de ce chunk
        sont repris au début du suivant. La quantité réellement réutilisée
        est enregistrée dans overlap_token_count.
        """

        assemblies: list[ChunkAssembly] = []

        current_units: list[RecursiveUnit] = []
        current_text_parts: list[str] = []
        current_overlap_token_count = 0

        for unit in units:
            candidate_parts = [*current_text_parts, unit.text]
            candidate_text = self._combine_parts(candidate_parts)
            candidate_token_count = self.tokenizer.count_tokens(
                candidate_text
            )

            if (
                current_text_parts
                and candidate_token_count > self.config.chunk_size
            ):
                completed_text = self._combine_parts(current_text_parts)

                assemblies.append(
                    self._create_assembly(
                        text=completed_text,
                        units=current_units,
                        overlap_token_count=current_overlap_token_count,
                    )
                )

                overlap_text = self._get_overlap_text(completed_text)

                current_units = []
                current_text_parts = []
                current_overlap_token_count = 0

                if overlap_text:
                    text_with_overlap = self._combine_parts(
                        [overlap_text, unit.text]
                    )

                    if (
                        self.tokenizer.count_tokens(text_with_overlap)
                        <= self.config.chunk_size
                    ):
                        current_text_parts.append(overlap_text)
                        current_overlap_token_count = (
                            self.tokenizer.count_tokens(overlap_text)
                        )

                current_units.append(unit)
                current_text_parts.append(unit.text)

                current_text = self._combine_parts(current_text_parts)
                current_token_count = self.tokenizer.count_tokens(
                    current_text
                )

                if current_token_count > self.config.chunk_size:
                    raise ChunkingError(
                        "Erreur interne : le nouveau chunk dépasse "
                        f"chunk_size ({current_token_count} > "
                        f"{self.config.chunk_size})."
                    )

                if current_token_count == self.config.chunk_size:
                    assemblies.append(
                        self._create_assembly(
                            text=current_text,
                            units=current_units,
                            overlap_token_count=(
                                current_overlap_token_count
                            ),
                        )
                    )

                    current_units = []
                    current_text_parts = []
                    current_overlap_token_count = 0

                continue

            current_units.append(unit)
            current_text_parts.append(unit.text)

            if candidate_token_count == self.config.chunk_size:
                assemblies.append(
                    self._create_assembly(
                        text=candidate_text,
                        units=current_units,
                        overlap_token_count=current_overlap_token_count,
                    )
                )

                current_units = []
                current_text_parts = []
                current_overlap_token_count = 0

        if current_units and current_text_parts:
            assemblies.append(
                self._create_assembly(
                    text=self._combine_parts(current_text_parts),
                    units=current_units,
                    overlap_token_count=current_overlap_token_count,
                )
            )

        return self._remove_empty_or_duplicate_assemblies(assemblies)

    def _create_assembly(
        self,
        *,
        text: str,
        units: Sequence[RecursiveUnit],
        overlap_token_count: int = 0,
    ) -> ChunkAssembly:
        """
        Crée une assembly interne et valide sa taille.
        """

        normalized = text.strip()
        token_count = self.tokenizer.count_tokens(normalized)

        if token_count > self.config.chunk_size:
            raise ChunkingError(
                "Erreur interne : un chunk récursif dépasse "
                f"chunk_size ({token_count} > "
                f"{self.config.chunk_size})."
            )

        return ChunkAssembly(
            text=normalized,
            units=list(units),
            overlap_token_count=overlap_token_count,
        )

    def _get_overlap_text(self, previous_text: str) -> str:
        """
        Récupère les derniers tokens du chunk précédent.
        """

        overlap = self.config.chunk_overlap

        if overlap == 0:
            return ""

        return self.tokenizer.take_last_tokens(
            previous_text,
            overlap,
        )

    @staticmethod
    def _combine_parts(
        parts: Sequence[str],
    ) -> str:
        """
        Combine les parties avec un espace unique.
        """

        return " ".join(
            part.strip()
            for part in parts
            if part.strip()
        ).strip()

    @staticmethod
    def _remove_empty_or_duplicate_assemblies(
        assemblies: Sequence[ChunkAssembly],
    ) -> list[ChunkAssembly]:
        """
        Supprime les chunks vides et les doublons consécutifs.
        """

        result: list[ChunkAssembly] = []
        previous_text: str | None = None

        for assembly in assemblies:
            normalized = assembly.text.strip()

            if not normalized:
                continue

            if normalized == previous_text:
                continue

            result.append(assembly)
            previous_text = normalized

        return result

    def _build_document_chunk(
        self,
        *,
        assembly: ChunkAssembly,
        chunk_index: int,
    ) -> DocumentChunk:
        """
        Transforme une ChunkAssembly en DocumentChunk.
        """

        covered_blocks = self._unique_blocks(
            unit.block for unit in assembly.units
        )

        if not covered_blocks:
            raise ChunkingError(
                "Impossible de déterminer les blocs couverts."
            )

        first_block = covered_blocks[0]

        section_ids = self._unique_non_empty(
            block.section_id for block in covered_blocks
        )

        section_headings = self._unique_non_empty(
            block.section_heading for block in covered_blocks
        )

        content_types = self._unique_non_empty(
            block.block_type for block in covered_blocks
        )

        heading_path = self._common_heading_path(
            [block.heading_path for block in covered_blocks]
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

        token_count = self.tokenizer.count_tokens(assembly.text)

        metadata = {
            "tokenizer_name": self.tokenizer.name,
            "covered_block_ids": [
                block.block_id for block in covered_blocks
            ],
            "covered_block_indexes": [
                block.block_index for block in covered_blocks
            ],
            "section_ids": section_ids,
            "section_headings": section_headings,
            "recursive_separators": list(self.separators),
            "overlap_token_count": assembly.overlap_token_count,
            "preserve_sentences": self.config.preserve_sentences,
            "preserve_code_blocks": self.config.preserve_code_blocks,
            "preserve_tables": self.config.preserve_tables,
        }

        return DocumentChunk.create(
            document_id=first_block.document_id,
            text=assembly.text,
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

    def _resolve_blocks(self, document: Any) -> list[TextBlock]:
        """
        Accepte une séquence de TextBlock ou un document complet.
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

    @staticmethod
    def _validate_blocks(
        blocks: Sequence[TextBlock],
    ) -> list[TextBlock]:
        """
        Vérifie que tous les blocs sont valides et appartiennent au même document.
        """

        if not isinstance(blocks, Sequence):
            raise ChunkingError(
                "blocks doit être une séquence de TextBlock."
            )

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

        document_ids = {
            block.document_id for block in validated
        }

        if len(document_ids) != 1:
            raise ChunkingError(
                "Tous les blocs doivent appartenir au même document."
            )

        return validated

    @staticmethod
    def _unique_blocks(
        blocks: Iterable[TextBlock],
    ) -> list[TextBlock]:
        """
        Retourne les blocs uniques dans leur ordre d'apparition.
        """

        result: list[TextBlock] = []
        block_ids: set[str] = set()

        for block in blocks:
            if block.block_id in block_ids:
                continue

            block_ids.add(block.block_id)
            result.append(block)

        return result

    @staticmethod
    def _unique_non_empty(
        values: Iterable[Any],
    ) -> list[str]:
        """
        Retourne les valeurs uniques et non vides.
        """

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
        Calcule le chemin de titres commun à tous les blocs du chunk.
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

    def _create_default_tokenizer(self) -> BaseTokenizer:
        """
        Crée le tokenizer défini dans ChunkingConfig.
        """

        tokenizer_name = self.config.tokenizer_name.strip()

        if tokenizer_name in {
            "",
            "default",
            "whitespace",
        }:
            return WhitespaceTokenizer()

        return HuggingFaceTokenizer(tokenizer_name)