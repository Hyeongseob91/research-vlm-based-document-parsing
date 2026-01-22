"""
Semantic Chunking Module for VLM Document Parsing Evaluation

This module provides text chunking strategies for creating semantic chunks
from parsed document content. All chunking parameters are controlled to
ensure fair comparison between different parsers.
"""

from .chunker import (
    ChunkingStrategy,
    ChunkerConfig,
    Chunk,
    TextChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    create_chunker,
)
from .metrics import (
    BoundaryScore,
    ChunkScore,
    ChunkingMetrics,
    calculate_boundary_score,
    calculate_chunk_score,
)

__all__ = [
    # Chunker classes
    "ChunkingStrategy",
    "ChunkerConfig",
    "Chunk",
    "TextChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "create_chunker",
    # Metrics
    "BoundaryScore",
    "ChunkScore",
    "ChunkingMetrics",
    "calculate_boundary_score",
    "calculate_chunk_score",
]
