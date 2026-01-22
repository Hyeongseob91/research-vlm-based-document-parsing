"""
Retrieval Evaluation Module for VLM Document Parsing

This module provides tools for evaluating retrieval performance:
- Embedding generation
- Similarity search
- Hit Rate and MRR calculation
- A/B comparison between parsers
"""

from .embedder import (
    EmbeddingConfig,
    TextEmbedder,
    create_embedder,
)
from .retriever import (
    RetrievalConfig,
    RetrievalResult,
    ChunkRetriever,
)
from .evaluator import (
    RetrievalMetrics,
    RetrievalEvaluator,
    compare_retrieval_performance,
)

__all__ = [
    # Embedding
    "EmbeddingConfig",
    "TextEmbedder",
    "create_embedder",
    # Retrieval
    "RetrievalConfig",
    "RetrievalResult",
    "ChunkRetriever",
    # Evaluation
    "RetrievalMetrics",
    "RetrievalEvaluator",
    "compare_retrieval_performance",
]
