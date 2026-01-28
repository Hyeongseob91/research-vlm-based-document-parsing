"""
Text Chunking Strategies

This module implements various text chunking strategies for document processing.
All chunkers use identical parameters to ensure fair comparison between parsers.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    RECURSIVE_CHARACTER = "recursive_character"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


@dataclass
class ChunkerConfig:
    """Configuration for text chunker."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_threshold: float = 0.9  # Semantic chunker breakpoint threshold
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    length_function: str = "character_count"  # character_count, token_count
    keep_separator: bool = True
    strip_whitespace: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "semantic_threshold": self.semantic_threshold,
            "separators": self.separators,
            "length_function": self.length_function,
            "keep_separator": self.keep_separator,
            "strip_whitespace": self.strip_whitespace,
        }


@dataclass
class Chunk:
    """Represents a single text chunk."""
    id: str
    content: str
    start_index: int
    end_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of chunk content."""
        return len(self.content)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "length": self.length,
            "metadata": self.metadata,
        }


class TextChunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, config: ChunkerConfig):
        self.config = config
        self._length_func = self._get_length_function()

    def _get_length_function(self) -> Callable[[str], int]:
        """Get the appropriate length function."""
        if self.config.length_function == "character_count":
            return len
        elif self.config.length_function == "token_count":
            # Simple whitespace tokenization for token count
            return lambda text: len(text.split())
        else:
            return len

    @abstractmethod
    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into chunks."""
        pass

    def _create_chunk(
        self,
        content: str,
        start_index: int,
        chunk_index: int,
        document_id: str
    ) -> Chunk:
        """Create a chunk with proper metadata."""
        if self.config.strip_whitespace:
            content = content.strip()

        return Chunk(
            id=f"{document_id}_chunk_{chunk_index}",
            content=content,
            start_index=start_index,
            end_index=start_index + len(content),
            metadata={
                "document_id": document_id,
                "chunk_index": chunk_index,
                "strategy": self.config.strategy.value,
            }
        )


class RecursiveCharacterChunker(TextChunker):
    """
    Recursive character text splitter.

    Splits text by trying each separator in order, recursively splitting
    chunks that are too large until they fit within the size limit.
    """

    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into chunks using recursive character splitting."""
        splits = self._split_text(text, self.config.separators)
        chunks = self._merge_splits(splits, document_id)
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        final_chunks = []
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []

        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for split in splits:
            if self._length_func(split) <= self.config.chunk_size:
                good_splits.append(split)
            else:
                # Chunk is too big, need to recurse
                if good_splits:
                    merged = self._merge_small_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if new_separators:
                    # Try with next separator
                    other_splits = self._split_text(split, new_separators)
                    final_chunks.extend(other_splits)
                else:
                    # No more separators, force split by chunk size
                    final_chunks.extend(self._force_split(split))

        # Handle remaining good splits
        if good_splits:
            merged = self._merge_small_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_small_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits together up to chunk size."""
        merged = []
        current = []
        current_length = 0

        for split in splits:
            split_length = self._length_func(split)

            if current_length + split_length + len(separator) <= self.config.chunk_size:
                current.append(split)
                current_length += split_length + (len(separator) if current else 0)
            else:
                if current:
                    joined = separator.join(current) if self.config.keep_separator else " ".join(current)
                    merged.append(joined)
                current = [split]
                current_length = split_length

        if current:
            joined = separator.join(current) if self.config.keep_separator else " ".join(current)
            merged.append(joined)

        return merged

    def _force_split(self, text: str) -> list[str]:
        """Force split text into chunk-sized pieces."""
        chunks = []
        chunk_size = self.config.chunk_size

        for i in range(0, len(text), chunk_size - self.config.chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(text):
                break

        return chunks

    def _merge_splits(self, splits: list[str], document_id: str) -> list[Chunk]:
        """Convert split strings to Chunk objects with overlap handling."""
        chunks = []
        current_index = 0

        for i, split in enumerate(splits):
            if not split.strip():
                continue

            chunk = self._create_chunk(split, current_index, len(chunks), document_id)

            # Handle overlap: adjust start index for next chunk
            if self.config.chunk_overlap > 0 and len(split) > self.config.chunk_overlap:
                current_index += len(split) - self.config.chunk_overlap
            else:
                current_index += len(split)

            chunks.append(chunk)

        return chunks


class FixedSizeChunker(TextChunker):
    """Simple fixed-size text chunker."""

    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        i = 0
        chunk_index = 0
        while i < len(text):
            end = min(i + chunk_size, len(text))
            content = text[i:end]

            if content.strip():
                chunk = self._create_chunk(content, i, chunk_index, document_id)
                chunks.append(chunk)
                chunk_index += 1

            # Move to next position with overlap
            i += chunk_size - overlap
            if i >= len(text):
                break

        return chunks


class SemanticChunker(TextChunker):
    """
    Semantic chunker that identifies topic boundaries.

    Uses embedding similarity to detect semantic shifts.
    Note: Requires sentence-transformers for embeddings.
    """

    def __init__(self, config: ChunkerConfig, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        super().__init__(config)
        self.embedding_model = embedding_model
        self._embedder = None

    def _get_embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic chunking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder

    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into semantic chunks based on topic boundaries."""
        # First split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._create_chunk(text, 0, 0, document_id)]

        # Get embeddings and find breakpoints
        embedder = self._get_embedder()
        embeddings = embedder.encode(sentences, convert_to_numpy=True)

        # Calculate similarity between consecutive sentences
        breakpoints = self._find_breakpoints(embeddings, threshold=self.config.semantic_threshold)

        # Create chunks at breakpoints
        chunks = []
        current_start = 0
        current_sentences = []
        current_index = 0

        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_text = " ".join(current_sentences)

            # Check if we should break here
            should_break = (
                i in breakpoints or
                self._length_func(current_text) >= self.config.chunk_size
            )

            if should_break and current_sentences:
                chunk = self._create_chunk(current_text, current_index, len(chunks), document_id)
                chunks.append(chunk)
                current_index += len(current_text)
                current_sentences = []

        # Add remaining sentences
        if current_sentences:
            chunk = self._create_chunk(
                " ".join(current_sentences),
                current_index,
                len(chunks),
                document_id
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting for Korean/English
        pattern = r'(?<=[.!?。])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(self, embeddings, threshold: float = 0.9) -> set[int]:
        """Find indices where semantic shifts occur."""
        import numpy as np

        breakpoints = set()
        for i in range(1, len(embeddings)):
            # Cosine similarity
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            if sim < (1 - threshold):  # Low similarity = topic change
                breakpoints.add(i)

        return breakpoints


class HierarchicalChunker(TextChunker):
    """
    Hierarchical chunker that respects document structure.

    Splits based on markdown headers and sections.
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into hierarchical chunks based on headers."""
        sections = self._split_by_headers(text)
        chunks = []

        for i, (level, title, content) in enumerate(sections):
            full_content = f"{'#' * level} {title}\n\n{content}" if title else content

            # If section is too large, use recursive splitter
            if self._length_func(full_content) > self.config.chunk_size:
                sub_chunker = RecursiveCharacterChunker(self.config)
                sub_chunks = sub_chunker.chunk(full_content, f"{document_id}_section_{i}")
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata["section_title"] = title
                    sub_chunk.metadata["section_level"] = level
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_chunk(full_content, 0, len(chunks), document_id)
                chunk.metadata["section_title"] = title
                chunk.metadata["section_level"] = level
                chunks.append(chunk)

        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[int, str, str]]:
        """Split text by markdown headers."""
        sections = []
        matches = list(self.HEADER_PATTERN.finditer(text))

        if not matches:
            return [(0, "", text)]

        # Content before first header
        if matches[0].start() > 0:
            sections.append((0, "", text[:matches[0].start()].strip()))

        # Process each header and its content
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Content is everything until next header
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            sections.append((level, title, content))

        return sections


def create_chunker(config: ChunkerConfig) -> TextChunker:
    """Factory function to create appropriate chunker."""
    chunkers = {
        ChunkingStrategy.FIXED: FixedSizeChunker,
        ChunkingStrategy.RECURSIVE_CHARACTER: RecursiveCharacterChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.HIERARCHICAL: HierarchicalChunker,
    }

    chunker_class = chunkers.get(config.strategy, RecursiveCharacterChunker)
    return chunker_class(config)
