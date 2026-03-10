"""Chunking quality metrics wrapper.

Wraps BC/CS metrics from wigtnocr.chunking.metrics for evaluation use.
"""

from wigtnocr.chunking.metrics import (
    calculate_bc,
    calculate_cs,
    evaluate_chunking,
    create_embedding_client,
    BCScore,
    CSScore,
    ChunkingMetrics,
)

__all__ = [
    "calculate_bc", "calculate_cs", "evaluate_chunking",
    "create_embedding_client",
    "BCScore", "CSScore", "ChunkingMetrics",
]
